// Most of the code for RAVG rendering stolen from https://github.com/servo/pathfinder

use {
    macroquad::{
        miniquad::{
            Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferId, BufferLayout,
            BufferSource, BufferType, BufferUsage, Equation, PassAction, Pipeline,
            RenderingBackend, ShaderMeta, TextureFormat, TextureId, TextureParams,
            UniformBlockLayout, UniformsSource, VertexAttribute, VertexFormat, VertexStep,
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
        collections::HashMap,
        f32::consts::{PI, SQRT_2},
        hash::Hash,
        mem,
    },
};

#[macro_use]
extern crate bitflags;

pub const PI_2: f32 = PI * 2.0;
const EPSILON: f32 = 0.001;

static QUAD_VERTEX_POSITIONS: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
static QUAD_VERTEX_INDICES: [u16; 6] = [0, 1, 3, 1, 2, 3];

const TEXTURE_METADATA_ENTRIES_PER_ROW: u32 = 128;
const TEXTURE_METADATA_TEXTURE_WIDTH: u32 = TEXTURE_METADATA_ENTRIES_PER_ROW;
const TEXTURE_METADATA_TEXTURE_HEIGHT: u32 = 65536 / TEXTURE_METADATA_ENTRIES_PER_ROW;

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

const FLATTENING_TOLERANCE: f32 = 0.25;

const MASK_TILES_ACROSS: u32 = 256;
const MASK_TILES_DOWN: u32 = 256;

const MASK_FRAMEBUFFER_WIDTH: u32 = TILE_WIDTH * MASK_TILES_ACROSS;
const MASK_FRAMEBUFFER_HEIGHT: u32 = TILE_HEIGHT / 4 * MASK_TILES_DOWN;

const MAX_FILLS_PER_BATCH: usize = 0x10000;

#[repr(C)]
pub struct FillUniforms {
    pub framebuffer_size: [f32; 2],
    pub tile_size: [f32; 2],
}

#[repr(C)]
pub struct TileUniforms {
    pub transform: Mat4,
    pub tile_size: [f32; 2],
    pub texture_metadata_size: [i32; 2],
    pub mask_texture_size0: [f32; 2],
}

#[derive(Clone)]
pub struct Path2D {
    outline: Outline,
    current_contour: Contour,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ArcDirection {
    CW,
    CCW,
}

impl Path2D {
    #[inline]
    pub fn new() -> Path2D {
        Path2D {
            outline: Outline::new(),
            current_contour: Contour::new(),
        }
    }

    #[inline]
    pub fn close_path(&mut self) {
        self.current_contour.close();
    }

    #[inline]
    pub fn move_to(&mut self, to: Vector2F) {
        self.flush_current_contour();
        self.current_contour.push_endpoint(to);
    }

    #[inline]
    pub fn line_to(&mut self, to: Vector2F) {
        self.current_contour.push_endpoint(to);
    }

    #[inline]
    pub fn quadratic_curve_to(&mut self, ctrl: Vector2F, to: Vector2F) {
        self.current_contour.push_quadratic(ctrl, to);
    }

    #[inline]
    pub fn bezier_curve_to(&mut self, ctrl0: Vector2F, ctrl1: Vector2F, to: Vector2F) {
        self.current_contour.push_cubic(ctrl0, ctrl1, to);
    }

    pub fn ellipse<A>(
        &mut self,
        center: Vector2F,
        axes: A,
        rotation: f32,
        start_angle: f32,
        end_angle: f32,
    ) where
        A: IntoVector2F,
    {
        self.flush_current_contour();

        let transform = Transform2F::from_scale(axes)
            .rotate(rotation)
            .translate(center);
        self.current_contour
            .push_arc(&transform, start_angle, end_angle, ArcDirection::CW);

        if end_angle - start_angle >= 2.0 * PI {
            self.current_contour.close();
        }
    }

    fn flush_current_contour(&mut self) {
        if !self.current_contour.is_empty() {
            self.outline
                .push_contour(mem::replace(&mut self.current_contour, Contour::new()));
        }
    }
}

#[derive(Clone, Debug)]
struct Outline {
    contours: Vec<Contour>,
    bounds: RectF,
}

impl Outline {
    #[inline]
    fn new() -> Outline {
        Outline {
            contours: vec![],
            bounds: RectF::default(),
        }
    }

    fn push_contour(&mut self, contour: Contour) {
        if contour.is_empty() {
            return;
        }

        if self.contours.is_empty() {
            self.bounds = contour.bounds;
        } else {
            self.bounds = self.bounds.union_rect(contour.bounds);
        }

        self.contours.push(contour);
    }

    fn transform(&mut self, transform: &Transform2F) {
        if transform.is_identity() {
            return;
        }

        let mut new_bounds = None;
        for contour in &mut self.contours {
            contour.transform(transform);
            contour.update_bounds(&mut new_bounds);
        }
        self.bounds = new_bounds.unwrap_or_else(RectF::default);
    }

    #[inline]
    fn close_all_contours(&mut self) {
        self.contours.iter_mut().for_each(|contour| contour.close());
    }
}

#[derive(Clone, Debug)]
struct Contour {
    points: Vec<Vector2F>,
    flags: Vec<PointFlags>,
    bounds: RectF,
    closed: bool,
}

impl Contour {
    #[inline]
    fn new() -> Contour {
        Contour {
            points: vec![],
            flags: vec![],
            bounds: RectF::default(),
            closed: false,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    #[inline]
    fn len(&self) -> u32 {
        self.points.len() as u32
    }

    #[inline]
    fn position_of(&self, index: u32) -> Vector2F {
        self.points[index as usize]
    }

    #[inline]
    fn iter(&self) -> ContourIter {
        ContourIter {
            contour: self,
            index: 1,
        }
    }

    #[inline]
    fn point_is_endpoint(&self, point_index: u32) -> bool {
        !self.flags[point_index as usize]
            .intersects(PointFlags::CONTROL_POINT_0 | PointFlags::CONTROL_POINT_1)
    }

    #[inline]
    fn close(&mut self) {
        self.closed = true;
    }

    fn push_arc(
        &mut self,
        transform: &Transform2F,
        start_angle: f32,
        end_angle: f32,
        direction: ArcDirection,
    ) {
        if end_angle - start_angle >= PI * 2.0 {
            self.push_ellipse(transform);
        } else {
            let start = vec2f(start_angle.cos(), start_angle.sin());
            let end = vec2f(end_angle.cos(), end_angle.sin());
            self.push_arc_from_unit_chord(transform, LineSegment2F::new(start, end), direction);
        }
    }

    fn push_arc_from_unit_chord(
        &mut self,
        transform: &Transform2F,
        mut chord: LineSegment2F,
        direction: ArcDirection,
    ) {
        let mut direction_transform = Transform2F::default();
        if direction == ArcDirection::CCW {
            chord *= vec2f(1.0, -1.0);
            direction_transform = Transform2F::from_scale(vec2f(1.0, -1.0));
        }

        let (mut vector, end_vector) = (UnitVector(chord.from()), UnitVector(chord.to()));
        for segment_index in 0..4 {
            let mut sweep_vector = end_vector.rev_rotate_by(vector);
            let last = sweep_vector.0.x() >= -EPSILON && sweep_vector.0.y() >= -EPSILON;

            let mut segment;
            if !last {
                sweep_vector = UnitVector(vec2f(0.0, 1.0));
                segment = Segment::quarter_circle_arc();
            } else {
                segment = Segment::arc_from_cos(sweep_vector.0.x());
            }

            let half_sweep_vector = sweep_vector.halve_angle();
            let rotation = Transform2F::from_rotation_vector(half_sweep_vector.rotate_by(vector));
            segment = segment.transform(&(*transform * direction_transform * rotation));

            let mut push_segment_flags = PushSegmentFlags::UPDATE_BOUNDS;
            if segment_index == 0 {
                push_segment_flags.insert(PushSegmentFlags::INCLUDE_FROM_POINT);
            }
            self.push_segment(&segment, push_segment_flags);

            if last {
                break;
            }

            vector = vector.rotate_by(sweep_vector);
        }

        const EPSILON: f32 = 0.001;
    }

    fn push_ellipse(&mut self, transform: &Transform2F) {
        let segment = Segment::quarter_circle_arc();
        let mut rotation;
        self.push_segment(
            &segment.transform(transform),
            PushSegmentFlags::UPDATE_BOUNDS | PushSegmentFlags::INCLUDE_FROM_POINT,
        );
        rotation = Transform2F::from_rotation_vector(UnitVector(vec2f(0.0, 1.0)));
        self.push_segment(
            &segment.transform(&(*transform * rotation)),
            PushSegmentFlags::UPDATE_BOUNDS,
        );
        rotation = Transform2F::from_rotation_vector(UnitVector(vec2f(-1.0, 0.0)));
        self.push_segment(
            &segment.transform(&(*transform * rotation)),
            PushSegmentFlags::UPDATE_BOUNDS,
        );
        rotation = Transform2F::from_rotation_vector(UnitVector(vec2f(0.0, -1.0)));
        self.push_segment(
            &segment.transform(&(*transform * rotation)),
            PushSegmentFlags::UPDATE_BOUNDS,
        );
    }

    #[inline]
    fn push_point(&mut self, point: Vector2F, flags: PointFlags, update_bounds: bool) {
        debug_assert!(!point.x().is_nan() && !point.y().is_nan());

        if update_bounds {
            let first = self.is_empty();
            union_rect(&mut self.bounds, point, first);
        }

        self.points.push(point);
        self.flags.push(flags);
    }

    #[inline]
    fn push_segment(&mut self, segment: &Segment, flags: PushSegmentFlags) {
        if segment.is_none() {
            return;
        }

        let update_bounds = flags.contains(PushSegmentFlags::UPDATE_BOUNDS);
        self.push_point(segment.baseline.from(), PointFlags::empty(), update_bounds);

        if !segment.is_line() {
            self.push_point(
                segment.ctrl.from(),
                PointFlags::CONTROL_POINT_0,
                update_bounds,
            );
            if !segment.is_quadratic() {
                self.push_point(
                    segment.ctrl.to(),
                    PointFlags::CONTROL_POINT_1,
                    update_bounds,
                );
            }
        }

        self.push_point(segment.baseline.to(), PointFlags::empty(), update_bounds);
    }

    #[inline]
    pub fn push_endpoint(&mut self, to: Vector2F) {
        self.push_point(to, PointFlags::empty(), true);
    }

    #[inline]
    pub fn push_quadratic(&mut self, ctrl: Vector2F, to: Vector2F) {
        self.push_point(ctrl, PointFlags::CONTROL_POINT_0, true);
        self.push_point(to, PointFlags::empty(), true);
    }

    #[inline]
    pub fn push_cubic(&mut self, ctrl0: Vector2F, ctrl1: Vector2F, to: Vector2F) {
        self.push_point(ctrl0, PointFlags::CONTROL_POINT_0, true);
        self.push_point(ctrl1, PointFlags::CONTROL_POINT_1, true);
        self.push_point(to, PointFlags::empty(), true);
    }

    fn transform(&mut self, transform: &Transform2F) {
        if transform.is_identity() {
            return;
        }

        for (point_index, point) in self.points.iter_mut().enumerate() {
            *point = *transform * *point;
            union_rect(&mut self.bounds, *point, point_index == 0);
        }
    }

    fn update_bounds(&self, bounds: &mut Option<RectF>) {
        *bounds = Some(match *bounds {
            None => self.bounds,
            Some(bounds) => bounds.union_rect(self.bounds),
        })
    }
}

struct ContourIter<'a> {
    contour: &'a Contour,
    index: u32,
}

impl Iterator for ContourIter<'_> {
    type Item = Segment;

    #[inline]
    fn next(&mut self) -> Option<Segment> {
        let contour = self.contour;

        let include_close_segment = self.contour.closed;
        if (self.index == contour.len() && !include_close_segment)
            || self.index == contour.len() + 1
        {
            return None;
        }

        let point0_index = self.index - 1;
        let point0 = contour.position_of(point0_index);
        if self.index == contour.len() {
            let point1 = contour.position_of(0);
            self.index += 1;
            return Some(Segment::line(LineSegment2F::new(point0, point1)));
        }

        let point1_index = self.index;
        self.index += 1;
        let point1 = contour.position_of(point1_index);
        if contour.point_is_endpoint(point1_index) {
            return Some(Segment::line(LineSegment2F::new(point0, point1)));
        }

        let point2_index = self.index;
        let point2 = contour.position_of(point2_index);
        self.index += 1;
        if contour.point_is_endpoint(point2_index) {
            return Some(Segment::quadratic(
                LineSegment2F::new(point0, point2),
                point1,
            ));
        }

        let point3_index = self.index;
        let point3 = contour.position_of(point3_index);
        self.index += 1;
        debug_assert!(contour.point_is_endpoint(point3_index));
        Some(Segment::cubic(
            LineSegment2F::new(point0, point3),
            LineSegment2F::new(point1, point2),
        ))
    }
}

bitflags! {
    struct PointFlags: u8 {
        const CONTROL_POINT_0 = 0x01;
        const CONTROL_POINT_1 = 0x02;
    }
}

#[inline]
fn union_rect(bounds: &mut RectF, new_point: Vector2F, first: bool) {
    if first {
        *bounds = RectF::from_points(new_point, new_point);
    } else {
        *bounds = bounds.union_point(new_point)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
enum SegmentKind {
    None,
    Line,
    Quadratic,
    Cubic,
}

bitflags! {
    struct SegmentFlags: u8 {
        const FIRST_IN_SUBPATH = 0x01;
        const CLOSES_SUBPATH = 0x02;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Segment {
    baseline: LineSegment2F,
    ctrl: LineSegment2F,
    kind: SegmentKind,
    flags: SegmentFlags,
}

impl Segment {
    #[inline]
    fn is_none(&self) -> bool {
        self.kind == SegmentKind::None
    }

    #[inline]
    fn is_line(&self) -> bool {
        self.kind == SegmentKind::Line
    }

    #[inline]
    fn is_quadratic(&self) -> bool {
        self.kind == SegmentKind::Quadratic
    }

    #[inline]
    fn is_cubic(&self) -> bool {
        self.kind == SegmentKind::Cubic
    }

    #[inline]
    fn line(line: LineSegment2F) -> Segment {
        Segment {
            baseline: line,
            ctrl: LineSegment2F::default(),
            kind: SegmentKind::Line,
            flags: SegmentFlags::empty(),
        }
    }

    #[inline]
    fn quadratic(baseline: LineSegment2F, ctrl: Vector2F) -> Segment {
        Segment {
            baseline,
            ctrl: LineSegment2F::new(ctrl, Vector2F::zero()),
            kind: SegmentKind::Quadratic,
            flags: SegmentFlags::empty(),
        }
    }

    #[inline]
    fn cubic(baseline: LineSegment2F, ctrl: LineSegment2F) -> Segment {
        Segment {
            baseline,
            ctrl,
            kind: SegmentKind::Cubic,
            flags: SegmentFlags::empty(),
        }
    }

    #[inline]
    fn quarter_circle_arc() -> Segment {
        let p0 = Vector2F::splat(SQRT_2 * 0.5);
        let p1 = vec2f(-SQRT_2 / 6.0 + 4.0 / 3.0, 7.0 * SQRT_2 / 6.0 - 4.0 / 3.0);
        let flip = vec2f(1.0, -1.0);
        let (p2, p3) = (p1 * flip, p0 * flip);
        Segment::cubic(LineSegment2F::new(p3, p0), LineSegment2F::new(p2, p1))
    }

    fn arc_from_cos(cos_sweep_angle: f32) -> Segment {
        if cos_sweep_angle >= 1.0 - EPSILON {
            return Segment::line(LineSegment2F::new(vec2f(1.0, 0.0), vec2f(1.0, 0.0)));
        }

        let term = F32x4::new(
            cos_sweep_angle,
            -cos_sweep_angle,
            cos_sweep_angle,
            -cos_sweep_angle,
        );
        let signs = F32x4::new(1.0, -1.0, 1.0, 1.0);
        let p3p0 = ((F32x4::splat(1.0) + term) * F32x4::splat(0.5)).sqrt() * signs;
        let (p0x, p0y) = (p3p0.z(), p3p0.w());
        let (p1x, p1y) = (4.0 - p0x, (1.0 - p0x) * (3.0 - p0x) / p0y);
        let p2p1 = F32x4::new(p1x, -p1y, p1x, p1y) * F32x4::splat(1.0 / 3.0);
        Segment::cubic(LineSegment2F(p3p0), LineSegment2F(p2p1))
    }

    #[inline]
    fn to_cubic(&self) -> Segment {
        if self.is_cubic() {
            return *self;
        }

        let mut new_segment = *self;
        let p1_2 = self.ctrl.from() + self.ctrl.from();
        new_segment.ctrl =
            LineSegment2F::new(self.baseline.from() + p1_2, p1_2 + self.baseline.to())
                * (1.0 / 3.0);
        new_segment.kind = SegmentKind::Cubic;
        new_segment
    }

    #[inline]
    fn split(&self, t: f32) -> (Segment, Segment) {
        if self.is_line() {
            let (before, after) = self.as_line_segment().split(t);
            (Segment::line(before), Segment::line(after))
        } else {
            self.to_cubic().as_cubic_segment().split(t)
        }
    }

    #[inline]
    fn as_cubic_segment(&self) -> CubicSegment {
        debug_assert!(self.is_cubic());
        CubicSegment(self)
    }

    #[inline]
    fn as_line_segment(&self) -> LineSegment2F {
        debug_assert!(self.is_line());
        self.baseline
    }

    #[inline]
    fn transform(self, transform: &Transform2F) -> Segment {
        Segment {
            baseline: *transform * self.baseline,
            ctrl: *transform * self.ctrl,
            kind: self.kind,
            flags: self.flags,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CubicSegment<'s>(&'s Segment);

impl CubicSegment<'_> {
    #[inline]
    fn is_flat(self, tolerance: f32) -> bool {
        let mut uv = F32x4::splat(3.0) * self.0.ctrl.0
            - self.0.baseline.0
            - self.0.baseline.0
            - self.0.baseline.reversed().0;
        uv = uv * uv;
        uv = uv.max(uv.zwxy());
        uv[0] + uv[1] <= 16.0 * tolerance * tolerance
    }

    #[inline]
    fn split(self, t: f32) -> (Segment, Segment) {
        let (baseline0, ctrl0, baseline1, ctrl1);
        if t <= 0.0 {
            let from = &self.0.baseline.from();
            baseline0 = LineSegment2F::new(*from, *from);
            ctrl0 = LineSegment2F::new(*from, *from);
            baseline1 = self.0.baseline;
            ctrl1 = self.0.ctrl;
        } else if t >= 1.0 {
            let to = &self.0.baseline.to();
            baseline0 = self.0.baseline;
            ctrl0 = self.0.ctrl;
            baseline1 = LineSegment2F::new(*to, *to);
            ctrl1 = LineSegment2F::new(*to, *to);
        } else {
            let tttt = F32x4::splat(t);

            let (p0p3, p1p2) = (self.0.baseline.0, self.0.ctrl.0);
            let p0p1 = p0p3.concat_xy_xy(p1p2);
            let p01p12 = p0p1 + tttt * (p1p2 - p0p1);
            let pxxp23 = p1p2 + tttt * (p0p3 - p1p2);
            let p12p23 = p01p12.concat_zw_zw(pxxp23);
            let p012p123 = p01p12 + tttt * (p12p23 - p01p12);
            let p123 = p012p123.zwzw();
            let p0123 = p012p123 + tttt * (p123 - p012p123);

            baseline0 = LineSegment2F(p0p3.concat_xy_xy(p0123));
            ctrl0 = LineSegment2F(p01p12.concat_xy_xy(p012p123));
            baseline1 = LineSegment2F(p0123.concat_xy_zw(p0p3));
            ctrl1 = LineSegment2F(p012p123.concat_zw_zw(p12p23));
        }

        (
            Segment {
                baseline: baseline0,
                ctrl: ctrl0,
                kind: SegmentKind::Cubic,
                flags: self.0.flags & SegmentFlags::FIRST_IN_SUBPATH,
            },
            Segment {
                baseline: baseline1,
                ctrl: ctrl1,
                kind: SegmentKind::Cubic,
                flags: self.0.flags & SegmentFlags::CLOSES_SUBPATH,
            },
        )
    }
}

bitflags! {
    struct PushSegmentFlags: u8 {
        const UPDATE_BOUNDS = 0x01;
        const INCLUDE_FROM_POINT = 0x02;
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct HashedColor(Color);

impl Hash for HashedColor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0.r as u8 * 255).hash(state);
        (self.0.g as u8 * 255).hash(state);
        (self.0.b as u8 * 255).hash(state);
        (self.0.a as u8 * 255).hash(state);
    }
}

impl Eq for HashedColor {}

/// The vector scene to be rendered.
#[derive(Clone, Default)]
pub struct Scene {
    pub paths: Vec<Path>,
    pub colors: Vec<Color>,
    pub cache: HashMap<HashedColor, PaintId>,
    pub bounds: RectF,
    pub view_box: RectF,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PaintId(u16);

#[derive(Clone, Debug)]
pub struct Path {
    outline: Outline,
    paint_id: PaintId,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Fill {
    line_segment: LineSegment2F,
    link: f32,
}

bitflags! {
    struct FramebufferFlags: u8 {
        const MASK_FRAMEBUFFER_IS_DIRTY = 0x01;
        const DEST_FRAMEBUFFER_IS_DIRTY = 0x02;
    }
}

struct MaskStorage {
    mask_img: TextureId,
    render_pass: miniquad::RenderPass,
    allocated_page_count: u32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
struct AlphaTileId([f32; 4]);

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Tile {
    tile_x: f32,
    tile_y: f32,
    alpha_tile_id: AlphaTileId,
    color: f32,
    ctrl: f32,
    backdrop: f32,
}

#[derive(Clone, Debug)]
struct BuiltPath {
    backdrops: Vec<i32>,
    tiles: Vec<Tile>,
    rect: RectI,
    tile_bounds: RectI,
}

fn round_rect_out_to_tile_bounds(rect: RectF) -> RectI {
    (rect * vec2f(1.0 / TILE_WIDTH as f32, 1.0 / TILE_HEIGHT as f32))
        .round_out()
        .to_i32()
}

struct Tiler<'a> {
    built_path: BuiltPath,
    fills: Vec<Fill>,
    outline: &'a Outline,
}

impl<'a> Tiler<'a> {
    fn new(outline: &'a Outline, view_box: RectF, paint_id: PaintId) -> Tiler<'a> {
        let bounds = outline.bounds.intersection(view_box).unwrap_or_default();
        let tile_bounds = round_rect_out_to_tile_bounds(bounds);

        let mut data =
            Vec::with_capacity(tile_bounds.size().x() as usize * tile_bounds.size().y() as usize);
        for y in tile_bounds.min_y()..tile_bounds.max_y() {
            for x in tile_bounds.min_x()..tile_bounds.max_x() {
                data.push(Tile {
                    tile_x: x as f32,
                    tile_y: y as f32,
                    alpha_tile_id: AlphaTileId((!0u32).to_le_bytes().map(|v| v as f32)),
                    color: paint_id.0 as f32,
                    backdrop: 0.0,
                    ctrl: 0.0,
                });
            }
        }

        Tiler {
            built_path: BuiltPath {
                backdrops: vec![0; tile_bounds.width() as usize],
                tiles: data,
                rect: tile_bounds,
                tile_bounds,
            },
            fills: vec![],
            outline,
        }
    }

    fn generate_fills(&mut self, view_box: RectF, next_alpha_tile_index: &mut usize) {
        for contour in &self.outline.contours {
            for segment in contour.iter() {
                process_segment(
                    &segment,
                    view_box,
                    next_alpha_tile_index,
                    &mut self.fills,
                    &mut self.built_path,
                );
            }
        }
    }

    fn prepare_tiles(&mut self) {
        let tiled_data = &mut self.built_path;
        let (backdrops, tiles) = (&mut tiled_data.backdrops, &mut tiled_data.tiles);
        let tiles_across = tiled_data.rect.width() as usize;
        for (draw_tile_index, draw_tile) in tiles.iter_mut().enumerate() {
            let column = draw_tile_index % tiles_across;
            let delta = draw_tile.backdrop as i32;
            draw_tile.backdrop = backdrops[column] as f32;

            backdrops[column] += delta;
        }
    }
}

fn process_segment(
    segment: &Segment,
    view_box: RectF,
    next_alpha_tile_index: &mut usize,
    fills: &mut Vec<Fill>,
    built_path: &mut BuiltPath,
) {
    if segment.is_quadratic() {
        let cubic = segment.to_cubic();
        return process_segment(&cubic, view_box, next_alpha_tile_index, fills, built_path);
    }

    if segment.is_line()
        || (segment.is_cubic() && segment.as_cubic_segment().is_flat(FLATTENING_TOLERANCE))
    {
        return process_line_segment(
            segment.baseline,
            view_box,
            next_alpha_tile_index,
            fills,
            built_path,
        );
    }
    let (prev, next) = segment.split(0.5);
    process_segment(&prev, view_box, next_alpha_tile_index, fills, built_path);
    process_segment(&next, view_box, next_alpha_tile_index, fills, built_path);
}

fn process_line_segment(
    line_segment: LineSegment2F,
    view_box: RectF,
    next_alpha_tile_index: &mut usize,
    fills: &mut Vec<Fill>,
    built_path: &mut BuiltPath,
) {
    let clip_box = RectF::from_points(
        vec2f(view_box.min_x(), f32::NEG_INFINITY),
        view_box.lower_right(),
    );
    let line_segment = match clip_line_segment_to_rect(line_segment, clip_box) {
        None => return,
        Some(line_segment) => line_segment,
    };

    let tile_size = vec2f(TILE_WIDTH as f32, TILE_HEIGHT as f32);
    let tile_size_recip = Vector2F::splat(1.0) / tile_size;

    let tile_line_segment = (line_segment.0 * tile_size_recip.0.concat_xy_xy(tile_size_recip.0))
        .floor()
        .to_i32x4();
    let from_tile_coords = Vector2I(tile_line_segment.xy());
    let to_tile_coords = Vector2I(tile_line_segment.zw());
    let vector = line_segment.vector();
    let vector_is_negative = vector.0.packed_lt(F32x2::default());
    let step = Vector2I((vector_is_negative | U32x2::splat(1)).to_i32x2());
    let first_tile_crossing =
        (from_tile_coords + Vector2I((!vector_is_negative & U32x2::splat(1)).to_i32x2())).to_f32()
            * tile_size;

    let mut t_max = (first_tile_crossing - line_segment.from()) / vector;
    let t_delta = (tile_size / vector).0.abs();

    let (mut current_position, mut tile_coords) = (line_segment.from(), from_tile_coords);
    let mut last_step_direction = None;

    loop {
        let next_step_direction = if t_max.x() < t_max.y() {
            StepDirection::X
        } else if t_max.x() > t_max.y() {
            StepDirection::Y
        } else if step.x() > 0 {
            StepDirection::X
        } else {
            StepDirection::Y
        };

        let next_t = (if next_step_direction == StepDirection::X {
            t_max.x()
        } else {
            t_max.y()
        })
        .min(1.0);
        let next_step_direction = if tile_coords == to_tile_coords {
            None
        } else {
            Some(next_step_direction)
        };

        let next_position = line_segment.sample(next_t);
        let clipped_line_segment = LineSegment2F::new(current_position, next_position);
        add_fill(
            fills,
            built_path,
            next_alpha_tile_index,
            clipped_line_segment,
            tile_coords,
        );
        if step.y() < 0 && next_step_direction == Some(StepDirection::Y) {
            let auxiliary_segment =
                LineSegment2F::new(clipped_line_segment.to(), tile_coords.to_f32() * tile_size);
            add_fill(
                fills,
                built_path,
                next_alpha_tile_index,
                auxiliary_segment,
                tile_coords,
            );
        } else if step.y() > 0 && last_step_direction == Some(StepDirection::Y) {
            let auxiliary_segment = LineSegment2F::new(
                tile_coords.to_f32() * tile_size,
                clipped_line_segment.from(),
            );
            add_fill(
                fills,
                built_path,
                next_alpha_tile_index,
                auxiliary_segment,
                tile_coords,
            );
        }
        if step.x() < 0 && last_step_direction == Some(StepDirection::X) {
            adjust_alpha_tile_backdrop(built_path, tile_coords, 1);
        } else if step.x() > 0 && next_step_direction == Some(StepDirection::X) {
            adjust_alpha_tile_backdrop(built_path, tile_coords, -1);
        }
        match next_step_direction {
            None => break,
            Some(StepDirection::X) => {
                if tile_coords.x() == to_tile_coords.x() {
                    break;
                }
                t_max += vec2f(t_delta.x(), 0.0);
                tile_coords += vec2i(step.x(), 0);
            }
            Some(StepDirection::Y) => {
                if tile_coords.y() == to_tile_coords.y() {
                    break;
                }
                t_max += vec2f(0.0, t_delta.y());
                tile_coords += vec2i(0, step.y());
            }
        }

        current_position = next_position;
        last_step_direction = next_step_direction;
    }
}

fn add_fill(
    fills: &mut Vec<Fill>,
    built_path: &mut BuiltPath,
    next_alpha_tile_index: &mut usize,
    segment: LineSegment2F,
    tile_coords: Vector2I,
) {
    if tile_coords_to_local_index(built_path, tile_coords).is_none() {
        return;
    }

    debug_assert_eq!(TILE_WIDTH, TILE_HEIGHT);
    let tile_size = F32x4::splat(TILE_WIDTH as f32);
    let tile_upper_left = tile_coords.to_f32().0.to_f32x4().xyxy() * tile_size;
    let segment = (segment.0 - tile_upper_left) * F32x4::splat(256.0);
    let (min, max) = (
        F32x4::default(),
        F32x4::splat((TILE_WIDTH * 256 - 1) as f32),
    );
    let segment = segment.clamp(min, max).to_i32x4();
    let (from_x, from_y, to_x, to_y) = (segment[0], segment[1], segment[2], segment[3]);
    if from_x == to_x {
        return;
    }
    let alpha_tile_id =
        get_or_allocate_alpha_tile_index(built_path, next_alpha_tile_index, tile_coords);
    fills.push(Fill {
        line_segment: LineSegment2F::new(
            Vector2F::new(from_x as f32, from_y as f32),
            Vector2F::new(to_x as f32, to_y as f32),
        ),
        link: {
            let alpha_tile_index = u32::from_le_bytes(alpha_tile_id.0.map(|v| v as u8));
            // dbg!(alpha_tile_index);
            alpha_tile_index as f32
        },
    });
}

#[inline]
fn tile_coords_to_local_index(built_path: &mut BuiltPath, coords: Vector2I) -> Option<u32> {
    if built_path.tile_bounds.contains_point(coords) {
        Some(tile_coords_to_local_index_unchecked(built_path, coords))
    } else {
        None
    }
}

fn get_or_allocate_alpha_tile_index(
    built_path: &mut BuiltPath,
    next_alpha_tile_index: &mut usize,
    tile_coords: Vector2I,
) -> AlphaTileId {
    let local_tile_index = tile_coords_to_local_index_unchecked(built_path, tile_coords) as usize;

    let tiles = &mut built_path.tiles;

    let alpha_tile_id = tiles[local_tile_index].alpha_tile_id;
    if u32::from_le_bytes(alpha_tile_id.0.map(|v| v as u8)) < !0 {
        return alpha_tile_id;
    }

    *next_alpha_tile_index += 1;
    let alpha_tile_index = *next_alpha_tile_index;
    let bytes_as_vec4 = ((alpha_tile_index) as u32).to_le_bytes().map(|v| v as f32);
    let new_alpha_tile_id = AlphaTileId(bytes_as_vec4);
    tiles[local_tile_index].alpha_tile_id = new_alpha_tile_id;
    new_alpha_tile_id
}

#[inline]
fn tile_coords_to_local_index_unchecked(built_path: &mut BuiltPath, coords: Vector2I) -> u32 {
    let tile_rect = built_path.tile_bounds;
    let offset = coords - tile_rect.origin();
    (offset.x() + tile_rect.width() * offset.y()) as u32
}

#[inline]
fn adjust_alpha_tile_backdrop(built_path: &mut BuiltPath, tile_coords: Vector2I, delta: i8) {
    let (tiles, backdrops) = (&mut built_path.tiles, &mut built_path.backdrops);

    let tile_offset = tile_coords - built_path.rect.origin();
    if tile_offset.x() < 0
        || tile_offset.x() >= built_path.rect.width()
        || tile_offset.y() >= built_path.rect.height()
    {
        return;
    }

    if tile_offset.y() < 0 {
        backdrops[tile_offset.x() as usize] += delta as i32;
        return;
    }

    let local_tile_index = coords_to_index_unchecked(built_path.rect, tile_coords);
    tiles[local_tile_index].backdrop += delta as f32;
}

#[inline]
fn coords_to_index_unchecked(rect: RectI, coords: Vector2I) -> usize {
    (coords.y() - rect.min_y()) as usize * rect.size().x() as usize
        + (coords.x() - rect.min_x()) as usize
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum StepDirection {
    X,
    Y,
}

bitflags! {
    struct Outcode: u8 {
        const LEFT   = 0x01;
        const RIGHT  = 0x02;
        const TOP    = 0x04;
        const BOTTOM = 0x08;
    }
}

fn compute_outcode(point: Vector2F, rect: RectF) -> Outcode {
    let mut outcode = Outcode::empty();
    if point.x() < rect.min_x() {
        outcode.insert(Outcode::LEFT);
    }
    if point.y() < rect.min_y() {
        outcode.insert(Outcode::TOP);
    }
    if point.x() > rect.max_x() {
        outcode.insert(Outcode::RIGHT);
    }
    if point.y() > rect.max_y() {
        outcode.insert(Outcode::BOTTOM);
    }
    outcode
}

fn clip_line_segment_to_rect(
    mut line_segment: LineSegment2F,
    rect: RectF,
) -> Option<LineSegment2F> {
    let mut outcode_from = compute_outcode(line_segment.from(), rect);
    let mut outcode_to = compute_outcode(line_segment.to(), rect);

    loop {
        if outcode_from.is_empty() && outcode_to.is_empty() {
            return Some(line_segment);
        }
        if !(outcode_from & outcode_to).is_empty() {
            return None;
        }

        let clip_from = outcode_from.bits() > outcode_to.bits();
        let (mut point, outcode) = if clip_from {
            (line_segment.from(), outcode_from)
        } else {
            (line_segment.to(), outcode_to)
        };

        if outcode.contains(Outcode::LEFT) {
            point = vec2f(
                rect.min_x(),
                lerp(
                    line_segment.from_y(),
                    line_segment.to_y(),
                    (rect.min_x() - line_segment.from_x())
                        / (line_segment.to_x() - line_segment.from_x()),
                ),
            );
        } else if outcode.contains(Outcode::RIGHT) {
            point = vec2f(
                rect.max_x(),
                lerp(
                    line_segment.from_y(),
                    line_segment.to_y(),
                    (rect.max_x() - line_segment.from_x())
                        / (line_segment.to_x() - line_segment.from_x()),
                ),
            );
        } else if outcode.contains(Outcode::TOP) {
            point = vec2f(
                lerp(
                    line_segment.from_x(),
                    line_segment.to_x(),
                    (rect.min_y() - line_segment.from_y())
                        / (line_segment.to_y() - line_segment.from_y()),
                ),
                rect.min_y(),
            );
        } else if outcode.contains(Outcode::BOTTOM) {
            point = vec2f(
                lerp(
                    line_segment.from_x(),
                    line_segment.to_x(),
                    (rect.max_y() - line_segment.from_y())
                        / (line_segment.to_y() - line_segment.from_y()),
                ),
                rect.max_y(),
            );
        }

        if clip_from {
            line_segment.set_from(point);
            outcode_from = compute_outcode(point, rect);
        } else {
            line_segment.set_to(point);
            outcode_to = compute_outcode(point, rect);
        }
    }
}

/// Main lib object that stores data necessary to render a scene.
pub struct Renderer<'a> {
    ctx: &'a mut dyn RenderingBackend,
    viewport: RectI,
    background_color: Color,
    texture_metadata_texture: TextureId,
    mask_storage: Option<MaskStorage>,
    alpha_tile_count: u32,
    framebuffer_flags: FramebufferFlags,
    fill_pipeline: Pipeline,
    fill_bindings: Bindings,
    tile_pipeline: Pipeline,
    tile_bindings: Bindings,
    tiles_vertex_indices_buffer: Option<BufferId>,
    tiles_vertex_indices_length: usize,
    buffered_fills: Vec<Fill>,
    pending_fills: Vec<Fill>,
}

impl<'a> Renderer<'a> {
    /// Creates a new renderer ready to render content
    pub fn new(ctx: &'a mut dyn RenderingBackend, framebuffer_size: (f32, f32), background_color: Color) -> Renderer<'a> {
        let viewport = RectI::new(
            Vector2I::default(),
            Vector2I::new(framebuffer_size.0 as i32, framebuffer_size.1 as i32),
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

        let image = image::load_from_memory_with_format(
            include_bytes!("../textures/area-lut.png"),
            image::ImageFormat::Png,
        )
        .unwrap();
        let image = image.to_rgba8();

        let area_lut_texture_id = ctx.new_texture_from_data_and_format(
            &image,
            TextureParams {
                width: 256 as u32,
                height: 256 as u32,
                format: TextureFormat::RGBA8,
                ..Default::default()
            },
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
            images: vec![area_lut_texture_id],
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
                            UniformDesc::new("uTextureMetadataSize", UniformType::Int2),
                            UniformDesc::new("uMaskTextureSize0", UniformType::Float2),
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
            images: vec![texture_metadata_texture, texture_metadata_texture],
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
                VertexAttribute::with_buffer("aMaskTexCoord0", VertexFormat::Float4, 1),
                VertexAttribute::with_buffer("aColor", VertexFormat::Float1, 1),
                VertexAttribute::with_buffer("aCtrlBackdrop", VertexFormat::Float2, 1),
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

            background_color,

            tiles_vertex_indices_buffer: None,
            tiles_vertex_indices_length: 0,

            texture_metadata_texture,
            mask_storage: None,
            alpha_tile_count: 0,
            framebuffer_flags: FramebufferFlags::empty(),

            fill_pipeline,
            fill_bindings,

            tile_pipeline,
            tile_bindings,

            buffered_fills: vec![],
            pending_fills: vec![],
        }
    }

    pub fn update_viewport(&mut self, framebuffer_size: (f32, f32)) {
        self.viewport = RectI::new(
            Vector2I::default(),
            Vector2I::new(framebuffer_size.0 as i32, framebuffer_size.1 as i32),
        );
    }

    pub fn render(&mut self, scene: Scene) {
        let transform = Transform2F::default();

        self.framebuffer_flags = FramebufferFlags::empty();
        // self.device.begin_commands();
        self.alpha_tile_count = 0;

        let mut next_alpha_tile_index = 0;

        let palette = scene.colors.clone();
        self.upload_palette(&palette);
        let mut built_paths = vec![];
        for path_object in &scene.paths {
            let mut outline = path_object.outline.clone();
            outline.close_all_contours();
            outline.transform(&transform);

            let paint_id = path_object.paint_id;

            let mut tiler = Tiler::new(&outline, scene.view_box, paint_id);

            tiler.generate_fills(scene.view_box, &mut next_alpha_tile_index);
            tiler.prepare_tiles();
            if !tiler.fills.is_empty() {
                self.add_fills(&tiler.fills, 0, tiler.fills.len());
            }

            built_paths.push(tiler.built_path);
        }

        self.flush_fills();

        let mut tiles = vec![];
        for cpu_data in &built_paths {
            for tile in &cpu_data.tiles {
                if tile.alpha_tile_id == AlphaTileId((!0u32).to_le_bytes().map(|v| v as f32))
                    && tile.backdrop == 0.0
                {
                    continue;
                }

                tiles.push(*tile);
            }
        }

        self.draw_tiles(&tiles);

        // self.allocator.purge_if_needed();
        // self.device.end_commands();
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

    fn add_fills(&mut self, added_fills: &[Fill], first_el: usize, last_el: usize) {
        if added_fills.is_empty() {
            return;
        }

        self.pending_fills.reserve(last_el - first_el);
        for fill in &added_fills[first_el..last_el] {
            self.alpha_tile_count = self.alpha_tile_count.max(fill.link as u32 + 1);
            self.pending_fills.push(*fill);
        }

        self.reallocate_alpha_tile_pages_if_necessary();

        if self.buffered_fills.len() + self.pending_fills.len() > MAX_FILLS_PER_BATCH {
            self.flush_fills();
        }

        self.buffered_fills.append(&mut self.pending_fills);
    }

    fn flush_fills(&mut self) {
        if self.buffered_fills.is_empty() {
            return;
        }

        debug_assert!(!self.buffered_fills.is_empty());
        debug_assert!(self.buffered_fills.len() <= u32::MAX as usize);

        let old_fill_buffer = self.fill_bindings.vertex_buffers[1];
        self.fill_bindings.vertex_buffers[1] = self.ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Dynamic,
            BufferSource::slice(&self.buffered_fills),
        );

        let fill_count = self.buffered_fills.len() as u32;
        self.buffered_fills.clear();

        self.draw_fills(fill_count);
        self.ctx.delete_buffer(self.fill_bindings.vertex_buffers[1]);
        self.fill_bindings.vertex_buffers[1] = old_fill_buffer;
    }

    fn draw_fills(&mut self, fill_count: u32) {
        let mask_viewport = self.mask_viewport().size().to_f32().0;
        let mask_storage = self
            .mask_storage
            .as_ref()
            .expect("Where's the mask storage?");

        let mut action = PassAction::Nothing;
        if !self
            .framebuffer_flags
            .contains(FramebufferFlags::MASK_FRAMEBUFFER_IS_DIRTY)
        {
            action = PassAction::clear_color(0.0, 0.0, 0.0, 0.0)
        };

        self.ctx.begin_pass(Some(mask_storage.render_pass), action);
        // self.ctx
        //     .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
        self.ctx.apply_pipeline(&self.fill_pipeline);
        self.ctx.apply_bindings(&self.fill_bindings);

        self.ctx
            .apply_uniforms(UniformsSource::table(&FillUniforms {
                framebuffer_size: [mask_viewport[0], mask_viewport[1]],
                tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
            }));
        self.ctx.draw(0, 6, fill_count as i32);
        self.ctx.end_render_pass();

        self.framebuffer_flags
            .insert(FramebufferFlags::MASK_FRAMEBUFFER_IS_DIRTY);
    }

    fn draw_tiles(&mut self, tiles: &Vec<Tile>) {
        if tiles.is_empty() {
            return;
        }

        let old_tile_vertex_buffer_id = self.tile_bindings.vertex_buffers[1];
        self.tile_bindings.vertex_buffers[1] = self.ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&tiles),
        );

        self.ensure_index_buffer(tiles.len());

        let clear_color = self.background_color;

        self.ctx.begin_default_pass(PassAction::clear_color(
            clear_color.r,
            clear_color.g,
            clear_color.b,
            clear_color.a,
        ));
        self.ctx.apply_pipeline(&self.tile_pipeline);
        self.ctx.apply_bindings(&self.tile_bindings);

        let transform = self.tile_transform().to_columns();
        let transform = transform
            .map(|v| (0..4).map(|i| v[i]).collect::<Vec<f32>>())
            .map(|v| Vec4::from_slice(&v));
        let transform = Mat4::from_cols(transform[0], transform[1], transform[2], transform[3]);
        let mask_storage = self.mask_storage.as_ref().unwrap();
        let texture_size = self.ctx.texture_size(mask_storage.mask_img);
        self.ctx
            .apply_uniforms(UniformsSource::table(&TileUniforms {
                transform,
                tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
                texture_metadata_size: [
                    TEXTURE_METADATA_TEXTURE_WIDTH.try_into().unwrap(),
                    TEXTURE_METADATA_TEXTURE_HEIGHT.try_into().unwrap(),
                ],
                mask_texture_size0: [texture_size.0 as f32, texture_size.1 as f32],
            }));
        self.ctx.draw(0, 6, tiles.len().try_into().unwrap());
        self.ctx.end_render_pass();

        self.ctx.delete_buffer(self.tile_bindings.vertex_buffers[1]);
        self.tile_bindings.vertex_buffers[1] = old_tile_vertex_buffer_id;
    }

    fn reallocate_alpha_tile_pages_if_necessary(&mut self) {
        let alpha_tile_pages_needed = (self.alpha_tile_count + 0xffff) >> 16;
        if let Some(ref mask_storage) = self.mask_storage {
            if alpha_tile_pages_needed <= mask_storage.allocated_page_count {
                return;
            }
        }

        let mask_img = self.ctx.new_render_texture(TextureParams {
            width: MASK_FRAMEBUFFER_WIDTH,
            height: MASK_FRAMEBUFFER_HEIGHT * alpha_tile_pages_needed,
            format: TextureFormat::RGBA16F,
            ..Default::default()
        });

        self.mask_storage = Some(MaskStorage {
            mask_img,
            render_pass: self.ctx.new_render_pass(mask_img, None),
            allocated_page_count: alpha_tile_pages_needed,
        });
        self.tile_bindings.images[1] = mask_img;
    }

    fn mask_viewport(&self) -> RectI {
        let page_count = match self.mask_storage {
            Some(ref mask_storage) => mask_storage.allocated_page_count as i32,
            None => 0,
        };
        let height = MASK_FRAMEBUFFER_HEIGHT as i32 * page_count;
        RectI::new(
            Vector2I::default(),
            vec2i(MASK_FRAMEBUFFER_WIDTH as i32, height),
        )
    }

    fn ensure_index_buffer(&mut self, mut length: usize) {
        length = length.next_power_of_two();
        if self.tiles_vertex_indices_length >= length {
            return;
        }
        let mut indices: Vec<u16> = Vec::with_capacity(length * 6);
        for index in 0..(length as u16) {
            indices.extend_from_slice(&[
                index * 4,
                index * 4 + 1,
                index * 4 + 2,
                index * 4 + 1,
                index * 4 + 3,
                index * 4 + 2,
            ]);
        }

        if let Some(tiles_vertex_indices_buffer) = self.tiles_vertex_indices_buffer.take() {
            self.ctx.delete_buffer(tiles_vertex_indices_buffer);
        }
        let tiles_vertex_indices_buffer = self.ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );
        self.tiles_vertex_indices_buffer = Some(tiles_vertex_indices_buffer);
        self.tiles_vertex_indices_length = length;
    }

    fn tile_transform(&self) -> Transform4F {
        let draw_viewport = self.viewport.size().to_f32();
        let scale = Vector4F::new(2.0 / draw_viewport.x(), -2.0 / draw_viewport.y(), 1.0, 1.0);
        Transform4F::from_scale(scale).translate(Vector4F::new(-1.0, 1.0, 0.0, 1.0))
    }
}

pub fn push_path(scene: &mut Scene, transform: &Transform2F, mut path: Path2D, color: &Color) {
    let paint_id = push_color(scene, color);
    path.flush_current_contour();
    let mut outline = path.outline;
    outline.transform(transform);
    let new_path_bounds = outline.bounds;
    scene.paths.push(Path { outline, paint_id });
    scene.bounds = scene.bounds.union_rect(new_path_bounds);
}

fn push_color(scene: &mut Scene, base_color: &Color) -> PaintId {
    if let Some(paint_id) = scene.cache.get(&HashedColor(*base_color)) {
        return *paint_id;
    }

    let paint_id = PaintId(scene.colors.len() as u16);
    scene.cache.insert(HashedColor(*base_color), paint_id);
    scene.colors.push(*base_color);
    paint_id
}
