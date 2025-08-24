// Most of the code for RAVG rendering stolen from https://github.com/servo/pathfinder

pub mod fileopen;
#[macro_use]
extern crate bitflags;

use fileopen::log_this;
use {
    macroquad::{
        miniquad::{
            Backend, Bindings, BlendFactor, BlendState, BlendValue, BufferLayout, BufferSource,
            BufferType, BufferUsage, Equation, PassAction, Pipeline, RenderingBackend, ShaderMeta,
            TextureFormat, TextureId, TextureParams, UniformBlockLayout, UniformsSource,
            VertexAttribute, VertexFormat, VertexStep,
        },
        prelude::*,
        ui::Id,
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

#[repr(C)]
pub struct FillUniforms {
    pub framebuffer_size: [f32; 2],
    pub tile_size: [f32; 2],
}

#[repr(C)]
pub struct MaskBackgroundUniforms {
    pub mask_size: [f32; 2],
    pub tile_size: [f32; 2],
    pub framebuffer_size: [f32; 2],
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
    pub id: Id,
    pub contours: Vec<Contour>,
    pub bounds: RectF,
    current_contour: isize,
    fills: Vec<Fill>,
    tiles: Vec<Tile>,
    used_mask_tiles: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArcDirection {
    CW,
    CCW,
}

impl Path2D {
    #[inline]
    pub fn new(id: Id) -> Path2D {
        Path2D {
            id,
            contours: vec![],
            bounds: RectF::default(),
            current_contour: -1,
            fills: vec![],
            tiles: vec![],
            used_mask_tiles: vec![],
        }
    }

    #[inline]
    pub fn close_path(&mut self) {
        self.contours[self.current_contour as usize].close();
    }

    #[inline]
    pub fn move_to(&mut self, to: Vector2F) {
        self.next_contour();
        self.contours[self.current_contour as usize].push_endpoint(to);
    }

    #[inline]
    pub fn line_to(&mut self, to: Vector2F) {
        self.contours[self.current_contour as usize].push_endpoint(to);
    }

    #[inline]
    pub fn quadratic_curve_to(&mut self, ctrl: Vector2F, to: Vector2F) {
        self.contours[self.current_contour as usize].push_quadratic(ctrl, to);
    }

    #[inline]
    pub fn bezier_curve_to(&mut self, ctrl0: Vector2F, ctrl1: Vector2F, to: Vector2F) {
        self.contours[self.current_contour as usize].push_cubic(ctrl0, ctrl1, to);
    }

    #[inline]
    pub fn arc(
        &mut self,
        center: Vector2F,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        direction: ArcDirection,
    ) {
        let transform = Transform2F::from_scale(radius).translate(center);
        self.contours[self.current_contour as usize].push_arc(
            &transform,
            start_angle,
            end_angle,
            direction,
        );
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
        self.next_contour();

        let transform = Transform2F::from_scale(axes)
            .rotate(rotation)
            .translate(center);
        self.contours[self.current_contour as usize].push_arc(
            &transform,
            start_angle,
            end_angle,
            ArcDirection::CW,
        );

        if end_angle - start_angle >= 2.0 * PI {
            self.contours[self.current_contour as usize].close();
        }
    }

    pub fn next_contour(&mut self) {
        if self.current_contour >= 0 {
            let prev_contour = &mut self.contours[self.current_contour as usize];
            if !prev_contour.is_empty() {
                self.bounds = self.bounds.union_rect(prev_contour.bounds);
            }
        }

        self.current_contour += 1;
        assert!(self.current_contour >= 0);
        let is_new = self.current_contour as usize == self.contours.len();
        if is_new {
            self.contours.push(Contour::new());
        }

        let contour = &mut self.contours[self.current_contour as usize];
        if contour.is_empty() {
            return;
        }

        self.bounds = self.bounds.union_rect(contour.bounds);

        if !is_new {
            contour.changed = false;
            contour.first_time = false;
            contour.cur_point = 0;
            contour.cur_flag = 0;
        }
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
pub struct Contour {
    cur_point: usize,
    cur_flag: usize,
    first_time: bool,
    changed: bool,
    points: Vec<Vector2F>,
    flags: Vec<PointFlags>,
    bounds: RectF,
    pub closed: bool,
}

impl Contour {
    #[inline]
    pub fn new() -> Contour {
        Contour {
            cur_point: 0,
            cur_flag: 0,
            first_time: true,
            changed: false,
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
    pub fn len(&self) -> u32 {
        self.points.len() as u32
    }

    #[inline]
    pub fn position_of(&self, index: u32) -> Vector2F {
        self.points[index as usize]
    }

    #[inline]
    pub fn position_of_last(&self, index: u32) -> Vector2F {
        self.points[self.points.len() - index as usize]
    }

    #[inline]
    pub fn iter(&self) -> ContourIter {
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

    pub fn push_arc_from_unit_chord(
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

        if !self.first_time {
            if !self.changed {
                if self.points[self.cur_point] != point {
                    self.changed = true;
                }
                self.cur_point += 1;
            }
        } else {
            self.changed = true;
            self.cur_point += 1;
        }

        if !self.changed {
            return;
        }

        if update_bounds {
            let first = self.is_empty();
            union_rect(&mut self.bounds, point, first);
        }

        let is_new = (self.cur_point - 1) == self.points.len();
        if is_new {
            self.points.push(point);
            self.flags.push(flags);
        } else if self.changed {
            self.points[self.cur_point - 1] = point;
        }
    }

    #[inline]
    pub fn push_segment(&mut self, segment: &Segment, flags: PushSegmentFlags) {
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

    pub fn update_bounds(&self, bounds: &mut Option<RectF>) {
        *bounds = Some(match *bounds {
            None => self.bounds,
            Some(bounds) => bounds.union_rect(self.bounds),
        })
    }
}

pub struct ContourIter<'a> {
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
    pub struct SegmentFlags: u8 {
        const FIRST_IN_SUBPATH = 0x01;
        const CLOSES_SUBPATH = 0x02;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub baseline: LineSegment2F,
    pub ctrl: LineSegment2F,
    kind: SegmentKind,
    pub flags: SegmentFlags,
}

impl Segment {
    #[inline]
    fn is_none(&self) -> bool {
        self.kind == SegmentKind::None
    }

    #[inline]
    pub fn is_line(&self) -> bool {
        self.kind == SegmentKind::Line
    }

    #[inline]
    pub fn is_quadratic(&self) -> bool {
        self.kind == SegmentKind::Quadratic
    }

    #[inline]
    pub fn is_cubic(&self) -> bool {
        self.kind == SegmentKind::Cubic
    }

    #[inline]
    pub fn line(line: LineSegment2F) -> Segment {
        Segment {
            baseline: line,
            ctrl: LineSegment2F::default(),
            kind: SegmentKind::Line,
            flags: SegmentFlags::empty(),
        }
    }

    #[inline]
    pub fn quadratic(baseline: LineSegment2F, ctrl: Vector2F) -> Segment {
        Segment {
            baseline,
            ctrl: LineSegment2F::new(ctrl, Vector2F::zero()),
            kind: SegmentKind::Quadratic,
            flags: SegmentFlags::empty(),
        }
    }

    #[inline]
    pub fn cubic(baseline: LineSegment2F, ctrl: LineSegment2F) -> Segment {
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
    pub fn split(&self, t: f32) -> (Segment, Segment) {
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

    pub fn arc_length(&self) -> f32 {
        // FIXME(pcwalton)
        self.baseline.vector().length()
    }

    pub fn time_for_distance(&self, distance: f32) -> f32 {
        // FIXME(pcwalton)
        distance / self.arc_length()
    }

    #[inline]
    pub fn sample(self, t: f32) -> Vector2F {
        // FIXME(pcwalton): Don't degree elevate!
        if self.is_line() {
            self.as_line_segment().sample(t)
        } else {
            self.to_cubic().as_cubic_segment().sample(t)
        }
    }

    #[inline]
    pub fn reversed(&self) -> Segment {
        Segment {
            baseline: self.baseline.reversed(),
            ctrl: if self.is_quadratic() {
                self.ctrl
            } else {
                self.ctrl.reversed()
            },
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

    #[inline]
    pub fn sample(self, t: f32) -> Vector2F {
        self.split(t).0.baseline.to()
    }
}

bitflags! {
    pub struct PushSegmentFlags: u8 {
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PaintId(u16);

#[derive(Clone, Debug)]
pub struct Path {
    path_id: Id,
    paint_id: PaintId,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Fill {
    line_segment: LineSegment2F,
    fill_index: f32,
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
#[repr(C)]
struct MaskTileId(f32);

impl MaskTileId {
    const INVALID: MaskTileId = MaskTileId(0xFFFFFF as u32 as f32);
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Tile {
    tile_x: f32,
    tile_y: f32,
    mask_tex_coord: MaskTileId,
    color: f32,
    backdrop: f32,
}

fn round_rect_out_to_tile_bounds(rect: RectF) -> RectI {
    (rect * vec2f(1.0 / TILE_WIDTH as f32, 1.0 / TILE_HEIGHT as f32))
        .round_out()
        .to_i32()
}

fn process_segment(
    segment: &Segment,
    view_box: RectF,
    next_mask_tile_index: &mut usize,
    used_mask_tiles: &mut Vec<(usize, usize)>,
    path_used_mask_tiles: &mut Vec<usize>,
    fills: &mut Vec<Fill>,
    backdrops: &mut Vec<i32>,
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
) {
    if segment.is_quadratic() {
        let cubic = segment.to_cubic();
        return process_segment(
            &cubic,
            view_box,
            next_mask_tile_index,
            used_mask_tiles,
            path_used_mask_tiles,
            fills,
            backdrops,
            tiles,
            path_tile_bounds,
        );
    }

    if segment.is_line()
        || (segment.is_cubic() && segment.as_cubic_segment().is_flat(FLATTENING_TOLERANCE))
    {
        return process_line_segment(
            segment.baseline,
            view_box,
            next_mask_tile_index,
            used_mask_tiles,
            path_used_mask_tiles,
            fills,
            backdrops,
            tiles,
            path_tile_bounds,
        );
    }
    let (prev, next) = segment.split(0.5);
    process_segment(
        &prev,
        view_box,
        next_mask_tile_index,
        used_mask_tiles,
        path_used_mask_tiles,
        fills,
        backdrops,
        tiles,
        path_tile_bounds,
    );
    process_segment(
        &next,
        view_box,
        next_mask_tile_index,
        used_mask_tiles,
        path_used_mask_tiles,
        fills,
        backdrops,
        tiles,
        path_tile_bounds,
    );
}

fn process_line_segment(
    line_segment: LineSegment2F,
    view_box: RectF,
    next_mask_tile_index: &mut usize,
    used_mask_tiles: &mut Vec<(usize, usize)>,
    path_used_mask_tiles: &mut Vec<usize>,
    fills: &mut Vec<Fill>,
    backdrops: &mut Vec<i32>,
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
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

    let mut current_position = line_segment.from();
    let mut tile_coords = from_tile_coords;
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
            tiles,
            path_tile_bounds,
            next_mask_tile_index,
            used_mask_tiles,
            path_used_mask_tiles,
            clipped_line_segment,
            tile_coords,
        );
        if step.y() < 0 && next_step_direction == Some(StepDirection::Y) {
            let auxiliary_segment =
                LineSegment2F::new(clipped_line_segment.to(), tile_coords.to_f32() * tile_size);
            add_fill(
                fills,
                tiles,
                path_tile_bounds,
                next_mask_tile_index,
                used_mask_tiles,
                path_used_mask_tiles,
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
                tiles,
                path_tile_bounds,
                next_mask_tile_index,
                used_mask_tiles,
                path_used_mask_tiles,
                auxiliary_segment,
                tile_coords,
            );
        }
        if step.x() < 0 && last_step_direction == Some(StepDirection::X) {
            adjust_mask_tile_backdrop(backdrops, tiles, path_tile_bounds, tile_coords, 1);
        } else if step.x() > 0 && next_step_direction == Some(StepDirection::X) {
            adjust_mask_tile_backdrop(backdrops, tiles, path_tile_bounds, tile_coords, -1);
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
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
    next_mask_tile_index: &mut usize,
    used_mask_tiles: &mut Vec<(usize, usize)>,
    path_used_mask_tiles: &mut Vec<usize>,
    segment: LineSegment2F,
    tile_coords: Vector2I,
) {
    if !path_tile_bounds.contains_point(tile_coords) {
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
    let mask_tile_id = get_or_allocate_mask_tile_index(
        tiles,
        path_tile_bounds,
        next_mask_tile_index,
        used_mask_tiles,
        path_used_mask_tiles,
        tile_coords,
    );
    fills.push(Fill {
        line_segment: LineSegment2F::new(
            Vector2F::new(from_x as f32, from_y as f32),
            Vector2F::new(to_x as f32, to_y as f32),
        ),
        fill_index: mask_tile_id.0,
    });
}

fn get_or_allocate_mask_tile_index(
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
    next_mask_tile_index: &mut usize,
    used_mask_tiles: &mut Vec<(usize, usize)>,
    path_used_mask_tiles: &mut Vec<usize>,
    tile_coords: Vector2I,
) -> MaskTileId {
    let offset = tile_coords - path_tile_bounds.origin();
    let local_tile_index = (offset.x() + path_tile_bounds.width() * offset.y()) as usize;

    if tiles[local_tile_index].mask_tex_coord != MaskTileId::INVALID {
        return tiles[local_tile_index].mask_tex_coord;
    }

    // TODO: Rewrite to be more effective
    let mut cur_used = 0;
    while cur_used < used_mask_tiles.len() {
        if *next_mask_tile_index + 1 < used_mask_tiles[cur_used].0 {
            *next_mask_tile_index += 1;
            path_used_mask_tiles.push(*next_mask_tile_index);
            break;
        } else if *next_mask_tile_index + 1 == used_mask_tiles[cur_used].0 {
            *next_mask_tile_index = used_mask_tiles[cur_used].1 + 1;
            path_used_mask_tiles.push(*next_mask_tile_index);
            cur_used += 1;
            break;
        } else if *next_mask_tile_index + 1 > used_mask_tiles[cur_used].1 {
            cur_used += 1;
        } else {
            unreachable!()
        }
    }
    if cur_used == used_mask_tiles.len() {
        *next_mask_tile_index += 1;
        path_used_mask_tiles.push(*next_mask_tile_index);
    }

    let new_mask_tile_id = MaskTileId(*next_mask_tile_index as f32);
    tiles[local_tile_index].mask_tex_coord = new_mask_tile_id;
    new_mask_tile_id
}

#[inline]
fn adjust_mask_tile_backdrop(
    backdrops: &mut Vec<i32>,
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
    tile_coords: Vector2I,
    delta: i8,
) {
    let tile_offset = tile_coords - path_tile_bounds.origin();
    if tile_offset.x() < 0
        || tile_offset.x() >= path_tile_bounds.width()
        || tile_offset.y() >= path_tile_bounds.height()
    {
        return;
    }

    if tile_offset.y() < 0 {
        backdrops[tile_offset.x() as usize] += delta as i32;
        return;
    }

    let local_tile_index = coords_to_index_unchecked(path_tile_bounds, tile_coords);
    tiles[local_tile_index].backdrop += delta as f32;
}

#[inline]
fn coords_to_index_unchecked(rect: &RectI, coords: Vector2I) -> usize {
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
    viewport: RectF,

    paths: Vec<Path>,
    colors: Vec<Color>,
    color_cache: HashMap<HashedColor, PaintId>,
    retained_paths: HashMap<Id, Path2D>,

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
    mask_to_screen: bool,
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

            paths: vec![],
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

            mask_to_screen: false,
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

    pub fn begin_path(&mut self, id: Id) -> &mut Path2D {
        match self.retained_paths.entry(id) {
            Entry::Occupied(entry) => {
                let path = entry.into_mut();
                path.current_contour = -1;
                path
            }
            Entry::Vacant(entry) => entry.insert(Path2D::new(id)),
        }
    }

    pub fn fill_path(&mut self, transform: &Transform2F, path_id: Id, color: &Color) {
        let paint_id = self.push_color(color);
        if let Entry::Occupied(path) = self.retained_paths.entry(path_id) {
            let path = path.into_mut();
            let changed = path.contours.iter().any(|contour| contour.changed);
            if changed {
                self.paths
                    .iter()
                    .position(|path| path.path_id == path_id)
                    .map(|e| self.paths.remove(e));
            }
            if !self.paths.iter().any(|path| path.path_id == path_id) {
                path.transform(transform);
                path.bounds = path
                    .bounds
                    .union_rect(path.contours[path.current_contour as usize].bounds);
                self.paths.push(Path { path_id, paint_id });
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
        for scene_path in &self.paths {
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

        if self.mask_background && !self.mask_to_screen {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx.apply_pipeline(&self.mask_background_pipeline);
            self.ctx.apply_bindings(&self.mask_background_bindings);

            let texture_size = self.ctx.texture_size(self.mask_img);
            self.ctx
                .apply_uniforms(UniformsSource::table(&MaskBackgroundUniforms {
                    mask_size: [texture_size.0 as f32, texture_size.1 as f32],
                    tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
                    framebuffer_size: [
                        MASK_FRAMEBUFFER_WIDTH as f32,
                        MASK_FRAMEBUFFER_HEIGHT as f32,
                    ],
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
        self.fills.clear();

        println!("DRAW FILLS!");

        if self.mask_to_screen {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
        } else {
            self.ctx.begin_pass(
                Some(self.mask_render_pass),
                PassAction::clear_color(0.0, 0.0, 0.0, 0.0),
            );
        }
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
        if self.mask_to_screen || !self.tiles_to_screen {
            return;
        }
        println!("DRAW TILES!: {}", self.tiles.len());

        let old_tile_vertex_buffer_id = self.tile_bindings.vertex_buffers[1];

        println!("SET TILES BUFFER!");
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
        let transform = Mat4::from_cols(transform[0], transform[1], transform[2], transform[3]);
        let texture_size = self.ctx.texture_size(self.mask_img);
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
        self.ctx.draw(0, 6, self.tiles.len().try_into().unwrap());
        self.ctx.end_render_pass();

        self.ctx.delete_buffer(self.tile_bindings.vertex_buffers[1]);
        self.tile_bindings.vertex_buffers[1] = old_tile_vertex_buffer_id;
    }

    fn tile_transform(&self) -> Transform4F {
        let viewport_size = self.viewport.size();
        let scale = Vector4F::new(2.0 / viewport_size.x(), -2.0 / viewport_size.y(), 1.0, 1.0);
        Transform4F::from_scale(scale).translate(Vector4F::new(-1.0, 1.0, 0.0, 1.0))
    }
}
