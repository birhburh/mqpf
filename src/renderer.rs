use {
    crate::{fileopen::log_this, Fill, Tile, MaskTileId, Path, TILE_WIDTH, TILE_HEIGHT, TILE_ZOOM, PaintId, HashedColor, ScenePath, Id, process_segment},
    macroquad::{
        miniquad::{
            Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferLayout, BufferSource,
            BufferType, BufferUsage, Equation, PassAction, Pipeline, RenderingBackend, ShaderMeta,
            TextureFormat, TextureId, TextureParams, UniformBlockLayout, UniformsSource,
            VertexAttribute, VertexFormat, VertexStep,
        },
        prelude::*,
    },
    pathfinder_geometry::{
        line_segment::LineSegment2F,
        rect::{RectF, RectI},
        transform2d::Transform2F,
        transform3d::Transform4F,
        unit_vector::UnitVector,
        util::{alignup_i32, lerp},
        vector::{vec2f, vec2i, IntoVector2F, Vector2F, Vector2I, Vector4F},
    },
    pathfinder_simd::default::{F32x2, F32x4, U32x2},
    std::{
        collections::{hash_map::Entry, HashMap},
        f32::consts::{PI, SQRT_2},
        hash::Hash,
    },
};

static QUAD_VERTEX_POSITIONS: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
static QUAD_VERTEX_INDICES: [u16; 6] = [0, 1, 3, 1, 2, 3];

const TEXTURE_METADATA_ENTRIES_PER_ROW: u32 = 128;
const TEXTURE_METADATA_TEXTURE_WIDTH: u32 = TEXTURE_METADATA_ENTRIES_PER_ROW;
const TEXTURE_METADATA_TEXTURE_HEIGHT: u32 = 65536 / TEXTURE_METADATA_ENTRIES_PER_ROW;

const MASK_TILES_ACROSS: u32 = 256;
const MASK_TILES_DOWN: u32 = 256;

const MASK_FRAMEBUFFER_WIDTH: u32 = TILE_WIDTH * MASK_TILES_ACROSS;
const MASK_FRAMEBUFFER_HEIGHT: u32 = TILE_HEIGHT / 4 * MASK_TILES_DOWN;


#[repr(C)]
pub struct FillUniforms {
    pub framebuffer_size: [f32; 2],
    pub tile_size: [f32; 2],
}

#[repr(C)]
pub struct MaskBackgroundUniforms {
    pub mask_size: [f32; 2],
    pub tile_size: [f32; 2],
    pub tile_zoom: f32,
    pub framebuffer_size: [f32; 2],
}

#[repr(C)]
pub struct TileUniforms {
    pub transform: Mat4,
    pub tile_size: [f32; 2],
    pub tile_zoom: f32,
    pub texture_metadata_size: [i32; 2],
    pub mask_texture_size0: [f32; 2],
    pub framebuffer_size: [f32; 2],
}

fn round_rect_out_to_tile_bounds(rect: RectF) -> RectI {
    (rect * vec2f(1.0 / TILE_WIDTH as f32, 1.0 / TILE_HEIGHT as f32))
        .round_out()
        .to_i32()
}

/// Main lib object that stores data necessary to render a scene.
pub struct Renderer<'a> {
    ctx: &'a mut dyn RenderingBackend,
    viewport: RectF,

    scene_paths: Vec<ScenePath>,
    colors: Vec<Color>,
    color_cache: HashMap<HashedColor, PaintId>,
    retained_paths: HashMap<Id, Path>,

    texture_metadata_texture: TextureId,
    _area_lut_texture: Texture2D,

    mask_img: TextureId,
    mask_render_pass: miniquad::RenderPass,

    fill_pipeline: Pipeline,
    fill_bindings: Bindings,
    mask_background_pipeline: Pipeline,
    mask_background_bindings: Bindings,
    tile_pipeline: Pipeline,
    tile_bindings: Bindings,

    used_mask_tiles: Vec<(usize, usize)>,

    fills: Vec<Fill>,
    tiles: Vec<Tile>,

    mask_background: bool,
    tiles_to_screen: bool,
}

impl<'a> Renderer<'a> {
    /// Creates a new renderer ready to render content
    pub fn new(ctx: &'a mut dyn RenderingBackend, framebuffer_size: (f32, f32)) -> Renderer<'a> {
        let viewport = RectF::new(
            vec2f(0.0, 0.0),
            vec2f(framebuffer_size.0, framebuffer_size.1),
        );

        let quad_vertex_positions_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&QUAD_VERTEX_POSITIONS),
        );
        let quad_vertex_indices_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&QUAD_VERTEX_INDICES),
        );

        let area_lut_texture = Texture2D::from_file_with_format(
            include_bytes!("../textures/area-lut.png"),
            Some(ImageFormat::Png),
        );

        let texture_metadata_texture = ctx.new_render_texture(TextureParams {
            width: TEXTURE_METADATA_TEXTURE_WIDTH,
            height: TEXTURE_METADATA_TEXTURE_HEIGHT,
            format: TextureFormat::RGBA8,
            ..Default::default()
        });

        let fill_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::empty::<Fill>(0),
        );

        let fill_shader = ctx
            .new_shader(
                match ctx.info().backend {
                    Backend::OpenGl => ShaderSource::Glsl {
                        vertex: include_str!("../shaders/fill.vs.glsl"),
                        fragment: include_str!("../shaders/fill.fs.glsl"),
                    },
                    Backend::Metal => todo!(),
                },
                ShaderMeta {
                    images: vec!["uAreaLUT".into()],
                    uniforms: UniformBlockLayout {
                        uniforms: vec![
                            UniformDesc::new("uFramebufferSize", UniformType::Float2),
                            UniformDesc::new("uTileSize", UniformType::Float2),
                        ],
                    },
                },
            )
            .unwrap();

        let fill_bindings = Bindings {
            vertex_buffers: vec![quad_vertex_positions_buffer, fill_buffer],
            index_buffer: quad_vertex_indices_buffer,
            images: vec![area_lut_texture.raw_miniquad_id()],
        };

        let fill_pipeline = ctx.new_pipeline(
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    stride: size_of::<Fill>() as i32,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("aTessCoord", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("aLineSegment", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("aTileIndex", VertexFormat::Float1, 1),
            ],
            fill_shader,
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::One,
                )),
                alpha_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::One,
                )),
                ..Default::default()
            },
        );

        let mask_img = ctx.new_render_texture(TextureParams {
            width: MASK_FRAMEBUFFER_WIDTH,
            height: MASK_FRAMEBUFFER_HEIGHT,
            format: TextureFormat::RGBA16F,
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let mask_render_pass = ctx.new_render_pass(mask_img, None);

        let mask_background_shader = ctx.new_shader(
            match ctx.info().backend {
                Backend::OpenGl => ShaderSource::Glsl {
                    vertex: include_str!("../shaders/mask_background.vs.glsl"),
                    fragment: include_str!("../shaders/mask_background.fs.glsl"),
                },
                Backend::Metal => todo!(),
            },
            ShaderMeta {
                images: vec!["uMaskTexture0".into()],
                uniforms: UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("uMaskTextureSize0", UniformType::Float2),
                        UniformDesc::new("uTileSize", UniformType::Float2),
                        UniformDesc::new("uTileZoom", UniformType::Float1),
                        UniformDesc::new("uFramebufferSize", UniformType::Float2),
                    ],
                },
            },
        );

        let mask_background_shader = match mask_background_shader {
            Ok(v) => Ok(v),
            Err(e) => {
                log_this(&format!("mask background shader error: \n{e}"));
                Err(e)
            }
        }
        .unwrap();
        let mask_background_bindings = Bindings {
            vertex_buffers: vec![quad_vertex_positions_buffer],
            index_buffer: quad_vertex_indices_buffer,
            images: vec![mask_img],
        };

        let mask_background_pipeline = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[VertexAttribute::new("in_pos", VertexFormat::Float2)],
            mask_background_shader,
            PipelineParams::default(),
        );

        let tile_shader = ctx
            .new_shader(
                match ctx.info().backend {
                    Backend::OpenGl => ShaderSource::Glsl {
                        vertex: include_str!("../shaders/tile.vs.glsl"),
                        fragment: include_str!("../shaders/tile.fs.glsl"),
                    },
                    Backend::Metal => todo!(),
                },
                ShaderMeta {
                    images: vec!["uTextureMetadata".into(), "uMaskTexture0".into()],
                    uniforms: UniformBlockLayout {
                        uniforms: vec![
                            UniformDesc::new("uTransform", UniformType::Mat4),
                            UniformDesc::new("uTileSize", UniformType::Float2),
                            UniformDesc::new("uTileZoom", UniformType::Float1),
                            UniformDesc::new("uTextureMetadataSize", UniformType::Int2),
                            UniformDesc::new("uMaskTextureSize0", UniformType::Float2),
                            UniformDesc::new("uFramebufferSize", UniformType::Float2),
                        ],
                    },
                },
            )
            .unwrap();

        let tile_vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::empty::<Tile>(0),
        );

        let tile_bindings = Bindings {
            vertex_buffers: vec![quad_vertex_positions_buffer, tile_vertex_buffer],
            index_buffer: quad_vertex_indices_buffer,
            images: vec![texture_metadata_texture, mask_img],
        };

        let tile_pipeline = ctx.new_pipeline(
            &[
                BufferLayout::default(),
                BufferLayout {
                    step_func: VertexStep::PerInstance,
                    stride: size_of::<Tile>() as i32,
                    ..Default::default()
                },
            ],
            &[
                VertexAttribute::with_buffer("aTileOffset", VertexFormat::Float2, 0),
                VertexAttribute::with_buffer("aTileOrigin", VertexFormat::Float2, 1),
                VertexAttribute::with_buffer("aMaskTexCoord", VertexFormat::Float1, 1),
                VertexAttribute::with_buffer("aColor", VertexFormat::Float1, 1),
                VertexAttribute::with_buffer("aCtrlBackdrop", VertexFormat::Float1, 1),
            ],
            tile_shader,
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                alpha_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::One,
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );

        Renderer {
            ctx,

            viewport,

            retained_paths: HashMap::new(),

            scene_paths: vec![],
            colors: vec![],
            color_cache: HashMap::default(),

            texture_metadata_texture,
            mask_img,
            mask_render_pass,
            _area_lut_texture: area_lut_texture,
            fill_pipeline,
            fill_bindings,
            mask_background_pipeline,
            mask_background_bindings,
            tile_pipeline,
            tile_bindings,

            used_mask_tiles: vec![],

            fills: vec![],
            tiles: vec![],

            mask_background: true,
            tiles_to_screen: true,
        }
    }

    pub fn update_viewport(&mut self, framebuffer_size: (f32, f32)) {
        self.viewport = RectF::new(
            vec2f(0.0, 0.0),
            vec2f(framebuffer_size.0, framebuffer_size.1),
        );
        self.retained_paths.clear();
    }

    pub fn begin_path(&mut self, id: Id) -> &mut Path {
        match self.retained_paths.entry(id) {
            Entry::Occupied(entry) => {
                let path = entry.into_mut();
                path.current_contour = -1;
                path
            }
            Entry::Vacant(entry) => entry.insert(Path::new(id)),
        }
    }

    pub fn fill_path(&mut self, path_id: Id, color: &Color) {
        let paint_id = self.push_color(color);
        if let Entry::Occupied(path) = self.retained_paths.entry(path_id) {
            let path = path.into_mut();
            let changed = path.contours.iter().any(|contour| contour.changed);
            if changed {
                self.scene_paths
                    .iter()
                    .position(|path| path.path_id == path_id)
                    .map(|e| self.scene_paths.remove(e));
            }
            if !self.scene_paths.iter().any(|path| path.path_id == path_id) {
                path.bounds = path
                    .bounds
                    .union_rect(path.contours[path.current_contour as usize].bounds);
                self.scene_paths.push(ScenePath { path_id, paint_id });
            }
        }
    }

    fn push_color(&mut self, base_color: &Color) -> PaintId {
        if let Some(paint_id) = self.color_cache.get(&HashedColor(*base_color)) {
            return *paint_id;
        }

        let paint_id = PaintId(self.colors.len() as u16);
        self.color_cache.insert(HashedColor(*base_color), paint_id);
        self.colors.push(*base_color);
        paint_id
    }

    pub fn render(&mut self) {
        let transform = Transform2F::default();

        let mut next_mask_tile_index = 0;
        let palette = self.colors.clone();
        self.upload_palette(&palette);

        let mut path_tiles = Vec::with_capacity(1000);
        let mut path_used_mask_tiles = Vec::with_capacity(1000);
        self.tiles.clear();
        self.fills.clear();
        self.used_mask_tiles.clear();
        for scene_path in &self.scene_paths {
            if let Entry::Occupied(path) = self.retained_paths.entry(scene_path.path_id) {
                let path = path.into_mut();
                path.close_all_contours();
                path.transform(&transform);

                let path_tile_bounds = round_rect_out_to_tile_bounds(path.bounds);

                for y in path_tile_bounds.min_y()..path_tile_bounds.max_y() {
                    for x in path_tile_bounds.min_x()..path_tile_bounds.max_x() {
                        path_tiles.push(Tile {
                            tile_x: x as f32,
                            tile_y: y as f32,
                            mask_tex_coord: MaskTileId::INVALID,
                            color: scene_path.paint_id.0 as f32,
                            backdrop: 0.0,
                        });
                    }
                }

                let mut path_fills = Vec::with_capacity(
                    path_tile_bounds.size().x() as usize * path_tile_bounds.size().y() as usize,
                );
                let mut backdrops = vec![0; path_tile_bounds.width() as usize];

                let mut changed = false;
                for contour in &path.contours {
                    if contour.changed {
                        changed = true;
                        for segment in contour.iter() {
                            process_segment(
                                &segment,
                                self.viewport,
                                &mut next_mask_tile_index,
                                &mut self.used_mask_tiles,
                                &mut path_used_mask_tiles,
                                &mut path_fills,
                                &mut backdrops,
                                &mut path_tiles,
                                &path_tile_bounds,
                            );
                        }
                    }
                }
                if changed {
                    let mut cur_used = 0;
                    let mut path_cur_used = 0;
                    // TODO: Rewrite to be more effective
                    while path_cur_used < path.used_mask_tiles.len() {
                        if path.used_mask_tiles[path_cur_used] == self.used_mask_tiles[cur_used].0 {
                            self.used_mask_tiles[cur_used].0 += 1;
                            if self.used_mask_tiles[cur_used].0 > self.used_mask_tiles[cur_used].1 {
                                self.used_mask_tiles.remove(cur_used);
                            }
                            path_cur_used += 1;
                        } else if path.used_mask_tiles[path_cur_used]
                            > self.used_mask_tiles[cur_used].0
                        {
                            if path.used_mask_tiles[path_cur_used]
                                <= self.used_mask_tiles[cur_used].1
                            {
                                self.used_mask_tiles.insert(
                                    cur_used + 1,
                                    (
                                        path.used_mask_tiles[path_cur_used] + 1,
                                        self.used_mask_tiles[cur_used].1,
                                    ),
                                );
                                self.used_mask_tiles[cur_used].1 =
                                    path.used_mask_tiles[path_cur_used] - 1;
                                cur_used += 1;
                                path_cur_used += 1;
                            } else {
                                path_cur_used += 1;
                            }
                        } else {
                            path_cur_used += 1;
                        }
                    }

                    path.used_mask_tiles = path_used_mask_tiles.clone();
                    path.tiles = path_tiles.clone();
                    path.fills = path_fills.clone();
                } else {
                    path_tiles = path.tiles.clone();
                    path_fills = path.fills.clone();
                    path_used_mask_tiles = path.used_mask_tiles.clone();
                }

                let mut cur_used = 0;
                let mut path_cur_used = 0;
                // TODO: Rewrite to be more effective
                while path_cur_used < path_used_mask_tiles.len() {
                    if cur_used < self.used_mask_tiles.len() {
                        if path_used_mask_tiles[path_cur_used] < self.used_mask_tiles[cur_used].0 {
                            self.used_mask_tiles.insert(
                                cur_used,
                                (
                                    path_used_mask_tiles[path_cur_used],
                                    path_used_mask_tiles[path_cur_used],
                                ),
                            );
                            path_cur_used += 1;
                        } else if path_used_mask_tiles[path_cur_used]
                            > self.used_mask_tiles[cur_used].0
                        {
                            if path_used_mask_tiles[path_cur_used]
                                <= self.used_mask_tiles[cur_used].1
                            {
                                unreachable!()
                            } else {
                                if path_used_mask_tiles[path_cur_used]
                                    == self.used_mask_tiles[cur_used].1 + 1
                                {
                                    self.used_mask_tiles[cur_used].1 =
                                        path_used_mask_tiles[path_cur_used];
                                    path_cur_used += 1;
                                } else {
                                    cur_used += 1;
                                }
                            }
                        } else {
                            unreachable!()
                        }

                        if cur_used + 1 < self.used_mask_tiles.len()
                            && self.used_mask_tiles[cur_used].1
                                == self.used_mask_tiles[cur_used + 1].0 - 1
                        {
                            self.used_mask_tiles[cur_used].1 = self.used_mask_tiles[cur_used + 1].1;
                            self.used_mask_tiles.remove(cur_used + 1);
                        }
                    } else {
                        self.used_mask_tiles.insert(
                            cur_used,
                            (
                                path_used_mask_tiles[path_cur_used],
                                path_used_mask_tiles[path_cur_used],
                            ),
                        );
                        path_cur_used += 1;
                    }
                }

                let tiles_across = path_tile_bounds.width() as usize;
                for (draw_tile_index, draw_tile) in path_tiles.iter_mut().enumerate() {
                    let column = draw_tile_index % tiles_across;
                    let delta = draw_tile.backdrop as i32;
                    draw_tile.backdrop = backdrops[column] as f32;

                    backdrops[column] += delta;
                }

                if !path_fills.is_empty() {
                    self.fills.append(&mut path_fills);
                }
                for tile in &path_tiles {
                    if tile.mask_tex_coord == MaskTileId::INVALID && tile.backdrop == 0.0 {
                        continue;
                    }

                    self.tiles.push(*tile);
                }
                path_tiles.resize(0, Tile::default());
                path_used_mask_tiles.resize(0, 0);
            }
        }

        self.draw_fills();

        if self.mask_background {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx.apply_pipeline(&self.mask_background_pipeline);
            self.ctx.apply_bindings(&self.mask_background_bindings);

            let texture_size = self.ctx.texture_size(self.mask_img);
            let viewport_size = self.viewport.size();
            self.ctx
                .apply_uniforms(UniformsSource::table(&MaskBackgroundUniforms {
                    mask_size: [texture_size.0 as f32, texture_size.1 as f32],
                    tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
                    tile_zoom: TILE_ZOOM,
                    framebuffer_size: [viewport_size.x(), viewport_size.y()],
                }));
            self.ctx.draw(0, 6, 1);
            self.ctx.end_render_pass();
        }

        self.draw_tiles();
    }

    fn upload_palette(&mut self, metadata: &Vec<Color>) {
        let entries_per_row = TEXTURE_METADATA_ENTRIES_PER_ROW.try_into().unwrap();
        let texture_width: i32 = TEXTURE_METADATA_TEXTURE_WIDTH.try_into().unwrap();
        let padded_texel_size =
            (alignup_i32(metadata.len() as i32, entries_per_row) * texture_width * 4) as usize;
        let mut texels = Vec::with_capacity(padded_texel_size);
        for base_color in metadata {
            let texel: [u8; 4] = (*base_color).into();
            texels.extend_from_slice(&texel);
        }
        while texels.len() < padded_texel_size {
            texels.push(u8::default())
        }

        let width = TEXTURE_METADATA_TEXTURE_WIDTH;
        let height = texels.len() as u32 / (4 * TEXTURE_METADATA_TEXTURE_WIDTH);
        self.ctx
            .texture_resize(self.texture_metadata_texture, width, height, Some(&texels));
    }

    fn draw_fills(&mut self) {
        if self.fills.is_empty() {
            return;
        }

        let old_fill_buffer = self.fill_bindings.vertex_buffers[1];
        self.fill_bindings.vertex_buffers[1] = self.ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Dynamic,
            BufferSource::slice(&self.fills),
        );

        let fill_count = self.fills.len() as u32;
        // log_this(&format!("fills: {:?}\n", &self.fills));
        // log_this(&format!("fill_count: {:?}\n", &fill_count));
        self.fills.clear();

        self.ctx.begin_pass(
            Some(self.mask_render_pass),
            PassAction::clear_color(0.0, 0.0, 0.0, 0.0),
        );
        self.ctx.apply_pipeline(&self.fill_pipeline);
        self.ctx.apply_bindings(&self.fill_bindings);

        self.ctx
            .apply_uniforms(UniformsSource::table(&FillUniforms {
                framebuffer_size: [
                    MASK_FRAMEBUFFER_WIDTH as f32,
                    MASK_FRAMEBUFFER_HEIGHT as f32,
                ],
                tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
            }));
        self.ctx.draw(0, 6, fill_count as i32);
        self.ctx.end_render_pass();

        self.ctx.delete_buffer(self.fill_bindings.vertex_buffers[1]);
        self.fill_bindings.vertex_buffers[1] = old_fill_buffer;
    }

    fn draw_tiles(&mut self) {
        if self.tiles.is_empty() {
            return;
        }
        if !self.tiles_to_screen {
            return;
        }
        println!("DRAW TILES!: {}", self.tiles.len());

        let old_tile_vertex_buffer_id = self.tile_bindings.vertex_buffers[1];

        println!("SET TILES BUFFER!");
        // log_this(&format!("TILES: {:?}\n", &self.tiles));
        self.tile_bindings.vertex_buffers[1] = self.ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&self.tiles),
        );

        self.ctx.begin_default_pass(PassAction::Nothing);
        self.ctx.apply_pipeline(&self.tile_pipeline);
        self.ctx.apply_bindings(&self.tile_bindings);

        let transform = self.tile_transform().to_columns();
        let transform = transform
            .map(|v| (0..4).map(|i| v[i]).collect::<Vec<f32>>())
            .map(|v| Vec4::from_slice(&v));
        // log_this(&format!("transform: {:?}\n", &transform));
        let transform = Mat4::from_cols(transform[0], transform[1], transform[2], transform[3]);
        let texture_size = self.ctx.texture_size(self.mask_img);
        let viewport_size = self.viewport.size();
        self.ctx
            .apply_uniforms(UniformsSource::table(&TileUniforms {
                transform,
                tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
                tile_zoom: TILE_ZOOM,
                texture_metadata_size: [
                    TEXTURE_METADATA_TEXTURE_WIDTH.try_into().unwrap(),
                    TEXTURE_METADATA_TEXTURE_HEIGHT.try_into().unwrap(),
                ],
                mask_texture_size0: [texture_size.0 as f32, texture_size.1 as f32],
                framebuffer_size: [viewport_size.x(), viewport_size.y()],
            }));
        self.ctx.draw(0, 6, self.tiles.len().try_into().unwrap());
        self.ctx.end_render_pass();

        self.ctx.delete_buffer(self.tile_bindings.vertex_buffers[1]);
        self.tile_bindings.vertex_buffers[1] = old_tile_vertex_buffer_id;
    }

    fn tile_transform(&self) -> Transform4F {
        let viewport_size = self.viewport.size();
        let scale = Vector4F::new(2.0 / viewport_size.x(), -2.0 / viewport_size.y(), 1.0, 1.0);

        // log_this(&format!("scale: {:?}\n", &scale));
        Transform4F::from_scale(scale).translate(Vector4F::new(-1.0, 1.0, 0.0, 1.0))
    }
}
