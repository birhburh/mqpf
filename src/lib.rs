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

#[cfg(feature = "svg")]
use {
    pathfinder_geometry::transform2d::Matrix2x2F,
    usvg::{
        tiny_skia_path::{PathSegment, Point},
        LineCap as UsvgLineCap, LineJoin as UsvgLineJoin,
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

#[cfg(feature = "svg")]
const HAIRLINE_STROKE_WIDTH: f32 = 0.0333;
#[cfg(feature = "svg")]
const TOLERANCE: f32 = 0.01;

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
    pub outline: Outline,
    current_contour: Contour,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArcDirection {
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

    #[inline]
    pub fn arc(
        &mut self,
        // center: Vec2,
        center: Vector2F,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        direction: ArcDirection,
    ) {
        // let transform = Affine2::from_scale_angle_translation(vec2(radius, radius), 0.0, center);
        let transform = Transform2F::from_scale(radius).translate(center);
        self.current_contour
            .push_arc(&transform, start_angle, end_angle, direction);
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
pub struct Outline {
    pub contours: Vec<Contour>,
    pub bounds: RectF,
}

impl Outline {
    #[inline]
    pub fn new() -> Outline {
        Outline {
            contours: vec![],
            bounds: RectF::default(),
        }
    }

    #[inline]
    pub fn from_segments<I>(segments: I) -> Outline
    where
        I: Iterator<Item = Segment>,
    {
        let mut outline = Outline::new();
        let mut current_contour = Contour::new();

        for segment in segments {
            if segment.flags.contains(SegmentFlags::FIRST_IN_SUBPATH) {
                if !current_contour.is_empty() {
                    outline
                        .contours
                        .push(mem::replace(&mut current_contour, Contour::new()));
                }
                current_contour.push_point(segment.baseline.from(), PointFlags::empty(), true);
            }

            if segment.flags.contains(SegmentFlags::CLOSES_SUBPATH) {
                if !current_contour.is_empty() {
                    current_contour.close();
                    let contour = mem::replace(&mut current_contour, Contour::new());
                    outline.push_contour(contour);
                }
                continue;
            }

            if segment.is_none() {
                continue;
            }

            if !segment.is_line() {
                current_contour.push_point(segment.ctrl.from(), PointFlags::CONTROL_POINT_0, true);
                if !segment.is_quadratic() {
                    current_contour.push_point(
                        segment.ctrl.to(),
                        PointFlags::CONTROL_POINT_1,
                        true,
                    );
                }
            }

            current_contour.push_point(segment.baseline.to(), PointFlags::empty(), true);
        }

        outline.push_contour(current_contour);
        outline
    }

    pub fn push_contour(&mut self, contour: Contour) {
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
pub struct Contour {
    points: Vec<Vector2F>,
    flags: Vec<PointFlags>,
    bounds: RectF,
    pub closed: bool,
}

impl Contour {
    #[inline]
    pub fn new() -> Contour {
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

        if update_bounds {
            let first = self.is_empty();
            union_rect(&mut self.bounds, point, first);
        }

        self.points.push(point);
        self.flags.push(flags);
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
struct AlphaTileId(f32);

impl AlphaTileId {
    const INVALID: AlphaTileId = AlphaTileId(0xFFFFFF as u32 as f32);
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct Tile {
    tile_x: f32,
    tile_y: f32,
    mask_tex_coord_0: AlphaTileId,
    mask_tex_coord_1: AlphaTileId,
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
    next_alpha_tile_index: &mut usize,
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
            next_alpha_tile_index,
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
            next_alpha_tile_index,
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
        next_alpha_tile_index,
        fills,
        backdrops,
        tiles,
        path_tile_bounds,
    );
    process_segment(
        &next,
        view_box,
        next_alpha_tile_index,
        fills,
        backdrops,
        tiles,
        path_tile_bounds,
    );
}

fn process_line_segment(
    line_segment: LineSegment2F,
    view_box: RectF,
    next_alpha_tile_index: &mut usize,
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
            tiles,
            path_tile_bounds,
            next_alpha_tile_index,
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
                tiles,
                path_tile_bounds,
                next_alpha_tile_index,
                auxiliary_segment,
                tile_coords,
            );
        }
        if step.x() < 0 && last_step_direction == Some(StepDirection::X) {
            adjust_alpha_tile_backdrop(backdrops, tiles, path_tile_bounds, tile_coords, 1);
        } else if step.x() > 0 && next_step_direction == Some(StepDirection::X) {
            adjust_alpha_tile_backdrop(backdrops, tiles, path_tile_bounds, tile_coords, -1);
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
    next_alpha_tile_index: &mut usize,
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
    let alpha_tile_id = get_or_allocate_alpha_tile_index(
        tiles,
        path_tile_bounds,
        next_alpha_tile_index,
        tile_coords,
    );
    fills.push(Fill {
        line_segment: LineSegment2F::new(
            Vector2F::new(from_x as f32, from_y as f32),
            Vector2F::new(to_x as f32, to_y as f32),
        ),
        link: alpha_tile_id.0,
    });
}

fn get_or_allocate_alpha_tile_index(
    tiles: &mut Vec<Tile>,
    path_tile_bounds: &RectI,
    next_alpha_tile_index: &mut usize,
    tile_coords: Vector2I,
) -> AlphaTileId {
    let offset = tile_coords - path_tile_bounds.origin();
    let local_tile_index = (offset.x() + path_tile_bounds.width() * offset.y()) as usize;

    if tiles[local_tile_index].mask_tex_coord_1.0 as u32 & 0xFF != 0xFF {
        return tiles[local_tile_index].mask_tex_coord_0;
    }

    *next_alpha_tile_index += 1;
    let new_alpha_tile_id = AlphaTileId(*next_alpha_tile_index as f32);
    tiles[local_tile_index].mask_tex_coord_0 = new_alpha_tile_id;
    tiles[local_tile_index].mask_tex_coord_1 = AlphaTileId(0.0);
    new_alpha_tile_id
}

#[inline]
fn adjust_alpha_tile_backdrop(
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
    viewport: RectI,
    background_color: Color,
    texture_metadata_texture: TextureId,
    mask_storage: Option<MaskStorage>,
    alpha_tile_count: u32,
    framebuffer_flags: FramebufferFlags,
    _area_lut_texture: Texture2D,
    fill_pipeline: Pipeline,
    fill_bindings: Bindings,
    mask_background_pipeline: Pipeline,
    mask_background_bindings: Bindings,
    tile_pipeline: Pipeline,
    tile_bindings: Bindings,
    tiles_vertex_indices_buffer: Option<BufferId>,
    tiles_vertex_indices_length: usize,
    buffered_fills: Vec<Fill>,
    pending_fills: Vec<Fill>,
    mask_background: bool,
    mask_to_screen: bool,
    tiles_to_screen: bool,
}

impl<'a> Renderer<'a> {
    /// Creates a new renderer ready to render content
    pub fn new(
        ctx: &'a mut dyn RenderingBackend,
        framebuffer_size: (f32, f32),
        background_color: Color,
    ) -> Renderer<'a> {
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

        let mask_background_shader = ctx
            .new_shader(
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
                        ],
                    },
                },
            )
            .unwrap();

        let mask_background_bindings = Bindings {
            vertex_buffers: vec![quad_vertex_positions_buffer],
            index_buffer: quad_vertex_indices_buffer,
            images: vec![texture_metadata_texture],
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
                VertexAttribute::with_buffer("aMaskTexCoord0", VertexFormat::Float1, 1),
                VertexAttribute::with_buffer("aMaskTexCoord1", VertexFormat::Float1, 1),
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

            background_color,

            tiles_vertex_indices_buffer: None,
            tiles_vertex_indices_length: 0,

            texture_metadata_texture,
            mask_storage: None,
            alpha_tile_count: 0,
            framebuffer_flags: FramebufferFlags::empty(),

            _area_lut_texture: area_lut_texture,
            fill_pipeline,
            fill_bindings,

            mask_background_pipeline,
            mask_background_bindings,

            tile_pipeline,
            tile_bindings,

            buffered_fills: vec![],
            pending_fills: vec![],
            mask_to_screen: false,
            mask_background: true,
            tiles_to_screen: true,
        }
    }

    pub fn update_viewport(&mut self, framebuffer_size: (f32, f32)) {
        self.viewport = RectI::new(
            Vector2I::default(),
            Vector2I::new(framebuffer_size.0 as i32, framebuffer_size.1 as i32),
        );
    }

    pub fn render(&mut self, scene: &Scene) {
        let transform = Transform2F::default();

        self.framebuffer_flags = FramebufferFlags::empty();
        self.alpha_tile_count = 0;

        let mut next_alpha_tile_index = 0;

        let palette = scene.colors.clone();
        self.upload_palette(&palette);
        let mut all_tiles = vec![];

        let mut tiles = Vec::with_capacity(1000);
        for path_object in &scene.paths {
            let mut outline = path_object.outline.clone();
            outline.close_all_contours();
            outline.transform(&transform);

            let bounds = outline
                .bounds
                .intersection(scene.view_box)
                .unwrap_or_default();
            let path_tile_bounds = round_rect_out_to_tile_bounds(bounds);

            for y in path_tile_bounds.min_y()..path_tile_bounds.max_y() {
                for x in path_tile_bounds.min_x()..path_tile_bounds.max_x() {
                    tiles.push(Tile {
                        tile_x: x as f32,
                        tile_y: y as f32,
                        mask_tex_coord_0: AlphaTileId::INVALID,
                        mask_tex_coord_1: AlphaTileId::INVALID,
                        color: path_object.paint_id.0 as f32,
                        backdrop: 0.0,
                    });
                }
            }

            let mut fills = Vec::with_capacity(
                path_tile_bounds.size().x() as usize * path_tile_bounds.size().y() as usize,
            );
            let mut backdrops = vec![0; path_tile_bounds.width() as usize];

            for contour in &outline.contours {
                for segment in contour.iter() {
                    process_segment(
                        &segment,
                        scene.view_box,
                        &mut next_alpha_tile_index,
                        &mut fills,
                        &mut backdrops,
                        &mut tiles,
                        &path_tile_bounds,
                    );
                }
            }

            let tiles_across = path_tile_bounds.width() as usize;
            for (draw_tile_index, draw_tile) in tiles.iter_mut().enumerate() {
                let column = draw_tile_index % tiles_across;
                let delta = draw_tile.backdrop as i32;
                draw_tile.backdrop = backdrops[column] as f32;

                backdrops[column] += delta;
            }

            if !fills.is_empty() {
                self.add_fills(&fills, 0, fills.len());
            }

            for tile in &tiles {
                if tile.mask_tex_coord_0 == AlphaTileId::INVALID && tile.backdrop == 0.0 {
                    continue;
                }

                all_tiles.push(*tile);
            }
            tiles.resize(0, Tile::default());
        }

        self.flush_fills();

        self.draw_tiles(&all_tiles);
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

        if self.mask_to_screen {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
        } else {
            self.ctx.begin_pass(Some(mask_storage.render_pass), action);
        }
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

        if self.mask_background && !self.mask_to_screen {
            self.ctx
                .begin_default_pass(PassAction::clear_color(0.0, 0.0, 0.0, 1.0));
            self.ctx.apply_pipeline(&self.mask_background_pipeline);
            self.ctx.apply_bindings(&self.mask_background_bindings);

            let texture_size = self.ctx.texture_size(mask_storage.mask_img);
            self.ctx
                .apply_uniforms(UniformsSource::table(&FillUniforms {
                    framebuffer_size: [texture_size.0 as f32, texture_size.1 as f32],
                    tile_size: [TILE_WIDTH as f32, TILE_HEIGHT as f32],
                }));
            self.ctx.draw(0, 6, 1);
            self.ctx.end_render_pass();
        }
    }

    fn draw_tiles(&mut self, tiles: &Vec<Tile>) {
        if tiles.is_empty() {
            return;
        }
        if self.mask_to_screen || !self.tiles_to_screen {
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
        let mut action = PassAction::Nothing;
        if !self.mask_background {
            action =
                PassAction::clear_color(clear_color.r, clear_color.g, clear_color.b, clear_color.a)
        };

        self.ctx.begin_default_pass(action);
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
        self.mask_background_bindings.images[0] = mask_img;
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

#[cfg(feature = "svg")]
struct UsvgPathToSegments<I>
where
    I: Iterator<Item = PathSegment>,
{
    iter: I,
    first_subpath_point: Vector2F,
    last_subpath_point: Vector2F,
    just_moved: bool,
}

#[cfg(feature = "svg")]
impl<I> UsvgPathToSegments<I>
where
    I: Iterator<Item = PathSegment>,
{
    fn new(iter: I) -> UsvgPathToSegments<I> {
        UsvgPathToSegments {
            iter,
            first_subpath_point: Vector2F::zero(),
            last_subpath_point: Vector2F::zero(),
            just_moved: false,
        }
    }
}

#[cfg(feature = "svg")]
impl<I> Iterator for UsvgPathToSegments<I>
where
    I: Iterator<Item = PathSegment>,
{
    type Item = Segment;

    fn next(&mut self) -> Option<Segment> {
        match self.iter.next()? {
            PathSegment::MoveTo(Point { x, y }) => {
                let to = vec2f(x as f32, y as f32);
                self.first_subpath_point = to;
                self.last_subpath_point = to;
                self.just_moved = true;
                self.next()
            }
            PathSegment::LineTo(Point { x, y }) => {
                let to = vec2f(x as f32, y as f32);
                let mut segment = Segment::line(LineSegment2F::new(self.last_subpath_point, to));
                if self.just_moved {
                    segment.flags.insert(SegmentFlags::FIRST_IN_SUBPATH);
                }
                self.last_subpath_point = to;
                self.just_moved = false;
                Some(segment)
            }
            PathSegment::CubicTo(
                Point { x: x1, y: y1 },
                Point { x: x2, y: y2 },
                Point { x, y },
            ) => {
                let ctrl0 = vec2f(x1 as f32, y1 as f32);
                let ctrl1 = vec2f(x2 as f32, y2 as f32);
                let to = vec2f(x as f32, y as f32);
                let mut segment = Segment::cubic(
                    LineSegment2F::new(self.last_subpath_point, to),
                    LineSegment2F::new(ctrl0, ctrl1),
                );
                if self.just_moved {
                    segment.flags.insert(SegmentFlags::FIRST_IN_SUBPATH);
                }
                self.last_subpath_point = to;
                self.just_moved = false;
                Some(segment)
            }
            PathSegment::QuadTo(Point { x: x1, y: y1 }, Point { x: x2, y: y2 }) => {
                let ctrl = vec2f(x1 as f32, y1 as f32);
                let to = vec2f(x2 as f32, y2 as f32);
                let mut segment =
                    Segment::quadratic(LineSegment2F::new(self.last_subpath_point, to), ctrl);
                if self.just_moved {
                    segment.flags.insert(SegmentFlags::FIRST_IN_SUBPATH);
                }
                self.last_subpath_point = to;
                self.just_moved = false;
                Some(segment)
            }
            PathSegment::Close => {
                let mut segment = Segment::line(LineSegment2F::new(
                    self.last_subpath_point,
                    self.first_subpath_point,
                ));
                segment.flags.insert(SegmentFlags::CLOSES_SUBPATH);
                self.just_moved = false;
                self.last_subpath_point = self.first_subpath_point;
                Some(segment)
            }
        }
    }
}

/// The shape of the ends of the stroke.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineCap {
    /// The ends of lines are squared off at the endpoints.
    Butt,
    /// The ends of lines are squared off by adding a box with an equal width and half the height
    /// of the line's thickness.
    Square,
    /// The ends of lines are rounded.
    Round,
}

#[cfg(feature = "svg")]
impl LineCap {
    #[inline]
    fn from_usvg_line_cap(usvg_line_cap: UsvgLineCap) -> LineCap {
        match usvg_line_cap {
            UsvgLineCap::Butt => LineCap::Butt,
            UsvgLineCap::Round => LineCap::Round,
            UsvgLineCap::Square => LineCap::Square,
        }
    }
}

#[cfg(feature = "svg")]
/// The shape used to join two line segments where they meet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineJoin {
    /// Connected segments are joined by extending their outside edges to connect at a single
    /// point, with the effect of filling an additional lozenge-shaped area. The `f32` value
    /// specifies the miter limit ratio.
    Miter(f32),
    /// Connected segments are joined by extending their outside edges to connect at a single
    /// point, with the effect of filling an additional lozenge-shaped area. The `f32` value
    /// specifies the miter limit ratio.
    MiterClip(f32),
    /// Fills an additional triangular area between the common endpoint of connected segments and
    /// the separate outside rectangular corners of each segment.
    Bevel,
    /// Rounds off the corners of a shape by filling an additional sector of disc centered at the
    /// common endpoint of connected segments. The radius for these rounded corners is equal to the
    /// line width.
    Round,
}

#[cfg(feature = "svg")]
impl LineJoin {
    #[inline]
    fn from_usvg_line_join(usvg_line_join: UsvgLineJoin, miter_limit: f32) -> LineJoin {
        match usvg_line_join {
            UsvgLineJoin::Miter => LineJoin::Miter(miter_limit),
            UsvgLineJoin::MiterClip => LineJoin::MiterClip(miter_limit),
            UsvgLineJoin::Round => LineJoin::Round,
            UsvgLineJoin::Bevel => LineJoin::Bevel,
        }
    }
}

#[cfg(feature = "svg")]
struct DashState<'a> {
    output: Contour,
    dashes: &'a [f32],
    current_dash_index: usize,
    distance_left: f32,
}

#[cfg(feature = "svg")]
impl<'a> DashState<'a> {
    fn new(dashes: &'a [f32], mut offset: f32) -> DashState<'a> {
        let total: f32 = dashes.iter().cloned().sum();
        offset %= total;

        let mut current_dash_index = 0;
        while current_dash_index < dashes.len() {
            let dash = dashes[current_dash_index];
            if offset < dash {
                break;
            }
            offset -= dash;
            current_dash_index += 1;
        }

        DashState {
            output: Contour::new(),
            dashes,
            current_dash_index,
            distance_left: offset,
        }
    }

    #[inline]
    fn is_on(&self) -> bool {
        self.current_dash_index % 2 == 0
    }
}

#[cfg(feature = "svg")]
struct ContourDash<'a, 'b, 'c> {
    input: &'a Contour,
    output: &'b mut Outline,
    state: &'c mut DashState<'a>,
}

#[cfg(feature = "svg")]
impl<'a, 'b, 'c> ContourDash<'a, 'b, 'c> {
    fn new(
        input: &'a Contour,
        output: &'b mut Outline,
        state: &'c mut DashState<'a>,
    ) -> ContourDash<'a, 'b, 'c> {
        ContourDash {
            input,
            output,
            state,
        }
    }

    fn dash(&mut self) {
        let mut iterator = self.input.iter();
        let mut queued_segment = None;
        loop {
            if queued_segment.is_none() {
                match iterator.next() {
                    None => break,
                    Some(segment) => queued_segment = Some(segment),
                }
            }

            let mut current_segment = queued_segment.take().unwrap();
            let mut distance = self.state.distance_left;

            let t = current_segment.time_for_distance(distance);
            if t < 1.0 {
                let (prev_segment, next_segment) = current_segment.split(t);
                current_segment = prev_segment;
                queued_segment = Some(next_segment);
            } else {
                distance = current_segment.arc_length();
            }

            if self.state.is_on() {
                self.state
                    .output
                    .push_segment(&current_segment, PushSegmentFlags::empty());
            }

            self.state.distance_left -= distance;
            if self.state.distance_left < EPSILON {
                if self.state.is_on() {
                    self.output
                        .push_contour(mem::replace(&mut self.state.output, Contour::new()));
                }

                self.state.current_dash_index += 1;
                if self.state.current_dash_index == self.state.dashes.len() {
                    self.state.current_dash_index = 0;
                }

                self.state.distance_left = self.state.dashes[self.state.current_dash_index];
            }
        }
    }
}

#[cfg(feature = "svg")]
/// Transforms a stroke into a dashed stroke.
pub struct OutlineDash<'a> {
    input: &'a Outline,
    output: Outline,
    state: DashState<'a>,
}

#[cfg(feature = "svg")]
impl<'a> OutlineDash<'a> {
    /// Creates a new outline dasher for the given stroke.
    ///
    /// Arguments:
    ///
    /// * `input`: The input stroke to be dashed. This must not yet been converted to a fill; i.e.
    ///   it is assumed that the stroke-to-fill conversion happens *after* this dashing process.
    ///
    /// * `dashes`: The list of dashes, specified as alternating pixel lengths of lines and gaps
    ///   that describe the pattern. See
    ///   <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/setLineDash>.
    ///
    /// * `offset`: The line dash offset, or "phase". See
    ///   <https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/lineDashOffset>.
    #[inline]
    pub fn new(input: &'a Outline, dashes: &'a [f32], offset: f32) -> OutlineDash<'a> {
        OutlineDash {
            input,
            output: Outline::new(),
            state: DashState::new(dashes, offset),
        }
    }

    /// Performs the dashing operation.
    ///
    /// The results can be retrieved with the `into_outline()` method.
    pub fn dash(&mut self) {
        for contour in &self.input.contours {
            ContourDash::new(contour, &mut self.output, &mut self.state).dash()
        }
    }

    /// Returns the resulting dashed outline.
    pub fn into_outline(mut self) -> Outline {
        if self.state.is_on() {
            self.output.push_contour(self.state.output);
        }
        self.output
    }
}

#[cfg(feature = "svg")]
/// How an outline should be stroked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokeStyle {
    /// The width of the stroke in scene units.
    pub line_width: f32,
    /// The shape of the ends of the stroke.
    pub line_cap: LineCap,
    /// The shape used to join two line segments where they meet.
    pub line_join: LineJoin,
}

#[cfg(feature = "svg")]
trait AddJoin {
    fn might_need_join(&self, join: LineJoin) -> bool;
    fn add_join(
        &mut self,
        distance: f32,
        join: LineJoin,
        join_point: Vector2F,
        next_tangent: LineSegment2F,
    );
}

#[cfg(feature = "svg")]
impl AddJoin for Contour {
    fn might_need_join(&self, join: LineJoin) -> bool {
        if self.len() < 2 {
            false
        } else {
            match join {
                LineJoin::Miter(_) | LineJoin::MiterClip(_) | LineJoin::Round => true,
                LineJoin::Bevel => false,
            }
        }
    }

    fn add_join(
        &mut self,
        distance: f32,
        join: LineJoin,
        join_point: Vector2F,
        next_tangent: LineSegment2F,
    ) {
        let (p0, p1) = (self.position_of_last(2), self.position_of_last(1));
        let prev_tangent = LineSegment2F::new(p0, p1);

        if prev_tangent.square_length() < EPSILON || next_tangent.square_length() < EPSILON {
            return;
        }

        match join {
            LineJoin::Bevel => {}
            LineJoin::Miter(miter_limit) | LineJoin::MiterClip(miter_limit) => {
                if let Some(prev_tangent_t) = prev_tangent.intersection_t(next_tangent) {
                    if prev_tangent_t < -EPSILON {
                        return;
                    }
                    let miter_endpoint = prev_tangent.sample(prev_tangent_t);
                    let threshold = miter_limit * distance;
                    if (miter_endpoint - join_point).square_length() > threshold * threshold {
                        return;
                    }
                    self.push_endpoint(miter_endpoint);
                }
            }
            LineJoin::Round => {
                let scale = distance.abs();
                let transform = Transform2F::from_scale(scale).translate(join_point);
                let chord_from = (prev_tangent.to() - join_point).normalize();
                let chord_to = (next_tangent.to() - join_point).normalize();
                let chord = LineSegment2F::new(chord_from, chord_to);
                self.push_arc_from_unit_chord(&transform, chord, ArcDirection::CW);
            }
        }
    }
}

#[cfg(feature = "svg")]
trait Offset {
    fn offset(&self, distance: f32, join: LineJoin, contour: &mut Contour);
    fn add_to_contour(
        &self,
        distance: f32,
        join: LineJoin,
        join_point: Vector2F,
        contour: &mut Contour,
    );
    fn offset_once(&self, distance: f32) -> Self;
    fn error_is_within_tolerance(&self, other: &Segment, distance: f32) -> bool;
}

#[cfg(feature = "svg")]
impl Offset for Segment {
    fn offset(&self, distance: f32, join: LineJoin, contour: &mut Contour) {
        let join_point = self.baseline.from();
        if self.baseline.square_length() < TOLERANCE * TOLERANCE {
            self.add_to_contour(distance, join, join_point, contour);
            return;
        }

        let candidate = self.offset_once(distance);
        if self.error_is_within_tolerance(&candidate, distance) {
            candidate.add_to_contour(distance, join, join_point, contour);
            return;
        }

        let (before, after) = self.split(0.5);
        before.offset(distance, join, contour);
        after.offset(distance, join, contour);
    }

    fn add_to_contour(
        &self,
        distance: f32,
        join: LineJoin,
        join_point: Vector2F,
        contour: &mut Contour,
    ) {
        // Add join if necessary.
        if contour.might_need_join(join) {
            let p3 = self.baseline.from();
            let p4 = if self.is_line() {
                self.baseline.to()
            } else {
                // NB: If you change the representation of quadratic curves, you will need to
                // change this.
                self.ctrl.from()
            };

            contour.add_join(distance, join, join_point, LineSegment2F::new(p4, p3));
        }

        // Push segment.
        let flags = PushSegmentFlags::UPDATE_BOUNDS | PushSegmentFlags::INCLUDE_FROM_POINT;
        contour.push_segment(self, flags);
    }

    fn offset_once(&self, distance: f32) -> Segment {
        if self.is_line() {
            return Segment::line(self.baseline.offset(distance));
        }

        if self.is_quadratic() {
            let mut segment_0 = LineSegment2F::new(self.baseline.from(), self.ctrl.from());
            let mut segment_1 = LineSegment2F::new(self.ctrl.from(), self.baseline.to());
            segment_0 = segment_0.offset(distance);
            segment_1 = segment_1.offset(distance);
            let ctrl = match segment_0.intersection_t(segment_1) {
                Some(t) => segment_0.sample(t),
                None => segment_0.to().lerp(segment_1.from(), 0.5),
            };
            let baseline = LineSegment2F::new(segment_0.from(), segment_1.to());
            return Segment::quadratic(baseline, ctrl);
        }

        debug_assert!(self.is_cubic());

        if self.baseline.from() == self.ctrl.from() {
            let mut segment_0 = LineSegment2F::new(self.baseline.from(), self.ctrl.to());
            let mut segment_1 = LineSegment2F::new(self.ctrl.to(), self.baseline.to());
            segment_0 = segment_0.offset(distance);
            segment_1 = segment_1.offset(distance);
            let ctrl = match segment_0.intersection_t(segment_1) {
                Some(t) => segment_0.sample(t),
                None => segment_0.to().lerp(segment_1.from(), 0.5),
            };
            let baseline = LineSegment2F::new(segment_0.from(), segment_1.to());
            let ctrl = LineSegment2F::new(segment_0.from(), ctrl);
            return Segment::cubic(baseline, ctrl);
        }

        if self.ctrl.to() == self.baseline.to() {
            let mut segment_0 = LineSegment2F::new(self.baseline.from(), self.ctrl.from());
            let mut segment_1 = LineSegment2F::new(self.ctrl.from(), self.baseline.to());
            segment_0 = segment_0.offset(distance);
            segment_1 = segment_1.offset(distance);
            let ctrl = match segment_0.intersection_t(segment_1) {
                Some(t) => segment_0.sample(t),
                None => segment_0.to().lerp(segment_1.from(), 0.5),
            };
            let baseline = LineSegment2F::new(segment_0.from(), segment_1.to());
            let ctrl = LineSegment2F::new(ctrl, segment_1.to());
            return Segment::cubic(baseline, ctrl);
        }

        let mut segment_0 = LineSegment2F::new(self.baseline.from(), self.ctrl.from());
        let mut segment_1 = LineSegment2F::new(self.ctrl.from(), self.ctrl.to());
        let mut segment_2 = LineSegment2F::new(self.ctrl.to(), self.baseline.to());
        segment_0 = segment_0.offset(distance);
        segment_1 = segment_1.offset(distance);
        segment_2 = segment_2.offset(distance);
        let (ctrl_0, ctrl_1) = match (
            segment_0.intersection_t(segment_1),
            segment_1.intersection_t(segment_2),
        ) {
            (Some(t0), Some(t1)) => (segment_0.sample(t0), segment_1.sample(t1)),
            _ => (
                segment_0.to().lerp(segment_1.from(), 0.5),
                segment_1.to().lerp(segment_2.from(), 0.5),
            ),
        };
        let baseline = LineSegment2F::new(segment_0.from(), segment_2.to());
        let ctrl = LineSegment2F::new(ctrl_0, ctrl_1);
        Segment::cubic(baseline, ctrl)
    }

    fn error_is_within_tolerance(&self, other: &Segment, distance: f32) -> bool {
        let (mut min, mut max) = (
            f32::abs(distance) - TOLERANCE,
            f32::abs(distance) + TOLERANCE,
        );
        min = if min <= 0.0 { 0.0 } else { min * min };
        max = if max <= 0.0 { 0.0 } else { max * max };

        for t_num in 0..(SAMPLE_COUNT + 1) {
            let t = t_num as f32 / SAMPLE_COUNT as f32;
            // FIXME(pcwalton): Use signed distance!
            let (this_p, other_p) = (self.sample(t), other.sample(t));
            let vector = this_p - other_p;
            let square_distance = vector.square_length();
            if square_distance < min || square_distance > max {
                return false;
            }
        }

        return true;

        const SAMPLE_COUNT: u32 = 16;
    }
}

#[cfg(feature = "svg")]
struct ContourStrokeToFill<'a> {
    input: &'a Contour,
    output: Contour,
    radius: f32,
    join: LineJoin,
}

#[cfg(feature = "svg")]
impl<'a> ContourStrokeToFill<'a> {
    #[inline]
    fn new(input: &Contour, output: Contour, radius: f32, join: LineJoin) -> ContourStrokeToFill {
        ContourStrokeToFill {
            input,
            output,
            radius,
            join,
        }
    }

    fn offset_forward(&mut self) {
        for (segment_index, segment) in self.input.iter().enumerate() {
            // FIXME(pcwalton): We negate the radius here so that round end caps can be drawn
            // clockwise. Of course, we should just implement anticlockwise arcs to begin with...
            let join = if segment_index == 0 {
                LineJoin::Bevel
            } else {
                self.join
            };
            segment.offset(-self.radius, join, &mut self.output);
        }
    }

    fn offset_backward(&mut self) {
        let mut segments: Vec<_> = self
            .input
            .iter()
            .map(|segment| segment.reversed())
            .collect();
        segments.reverse();
        for (segment_index, segment) in segments.iter().enumerate() {
            // FIXME(pcwalton): We negate the radius here so that round end caps can be drawn
            // clockwise. Of course, we should just implement anticlockwise arcs to begin with...
            let join = if segment_index == 0 {
                LineJoin::Bevel
            } else {
                self.join
            };
            segment.offset(-self.radius, join, &mut self.output);
        }
    }
}

#[cfg(feature = "svg")]
pub struct OutlineStrokeToFill<'a> {
    input: &'a Outline,
    output: Outline,
    style: StrokeStyle,
}

#[cfg(feature = "svg")]
impl<'a> OutlineStrokeToFill<'a> {
    /// Creates a new `OutlineStrokeToFill` object that will stroke the given outline with the
    /// given stroke style.
    #[inline]
    pub fn new(input: &Outline, style: StrokeStyle) -> OutlineStrokeToFill {
        OutlineStrokeToFill {
            input,
            output: Outline::new(),
            style,
        }
    }

    /// Performs the stroke operation.
    pub fn offset(&mut self) {
        let mut new_contours = vec![];
        for input in &self.input.contours {
            let closed = input.closed;
            let mut stroker = ContourStrokeToFill::new(
                input,
                Contour::new(),
                self.style.line_width * 0.5,
                self.style.line_join,
            );

            stroker.offset_forward();
            if closed {
                self.push_stroked_contour(&mut new_contours, stroker, true);
                stroker = ContourStrokeToFill::new(
                    input,
                    Contour::new(),
                    self.style.line_width * 0.5,
                    self.style.line_join,
                );
            } else {
                self.add_cap(&mut stroker.output);
            }

            stroker.offset_backward();
            if !closed {
                self.add_cap(&mut stroker.output);
            }

            self.push_stroked_contour(&mut new_contours, stroker, closed);
        }

        let mut new_bounds = None;
        new_contours
            .iter()
            .for_each(|contour| contour.update_bounds(&mut new_bounds));

        self.output.contours = new_contours;
        self.output.bounds = new_bounds.unwrap_or_default();
    }

    /// Returns the resulting stroked outline. This should be called after `offset()`.
    #[inline]
    pub fn into_outline(self) -> Outline {
        self.output
    }

    fn push_stroked_contour(
        &mut self,
        new_contours: &mut Vec<Contour>,
        mut stroker: ContourStrokeToFill,
        closed: bool,
    ) {
        // Add join if necessary.
        if closed && stroker.output.might_need_join(self.style.line_join) {
            let (p1, p0) = (stroker.output.position_of(1), stroker.output.position_of(0));
            let final_segment = LineSegment2F::new(p1, p0);
            stroker.output.add_join(
                self.style.line_width * 0.5,
                self.style.line_join,
                stroker.input.position_of(0),
                final_segment,
            );
        }

        stroker.output.closed = true;
        new_contours.push(stroker.output);
    }

    fn add_cap(&mut self, contour: &mut Contour) {
        if self.style.line_cap == LineCap::Butt || contour.len() < 2 {
            return;
        }

        let width = self.style.line_width;
        let p1 = contour.position_of_last(1);

        // Determine the ending gradient.
        let mut p0;
        let mut p0_index = contour.len() - 2;
        loop {
            p0 = contour.position_of(p0_index);
            if (p1 - p0).square_length() > EPSILON {
                break;
            }
            if p0_index == 0 {
                return;
            }
            p0_index -= 1;
        }
        let gradient = (p1 - p0).normalize();

        match self.style.line_cap {
            LineCap::Butt => unreachable!(),

            LineCap::Square => {
                let offset = gradient * (width * 0.5);

                let p2 = p1 + offset;
                let p3 = p2 + gradient.yx() * vec2f(-width, width);
                let p4 = p3 - offset;

                contour.push_endpoint(p2);
                contour.push_endpoint(p3);
                contour.push_endpoint(p4);
            }

            LineCap::Round => {
                let scale = width * 0.5;
                let offset = gradient.yx() * vec2f(-1.0, 1.0);
                let translation = p1 + offset * (width * 0.5);
                let transform = Transform2F::from_scale(scale).translate(translation);
                let chord = LineSegment2F::new(-offset, offset);
                contour.push_arc_from_unit_chord(&transform, chord, ArcDirection::CW);
            }
        }
    }
}

#[cfg(feature = "svg")]
fn render_node(node: &usvg::Node, scene: &mut Scene, global_transform: Transform2F) {
    match node {
        usvg::Node::Path(ref p) => {
            let t = node.abs_transform();
            let mut transform = global_transform;

            transform *= Transform2F {
                matrix: Matrix2x2F::row_major(t.sx, t.ky, t.kx, t.sy),
                vector: Vector2F::new(t.tx, t.ty),
            };

            let mut path = Path2D::new();
            for segment in p.data().segments() {
                match segment {
                    PathSegment::MoveTo(Point { x, y }) => {
                        path.move_to(vec2f(x as f32, y as f32));
                    }
                    PathSegment::LineTo(Point { x, y }) => {
                        path.line_to(vec2f(x as f32, y as f32));
                    }
                    PathSegment::CubicTo(
                        Point { x: x1, y: y1 },
                        Point { x: x2, y: y2 },
                        Point { x, y },
                    ) => {
                        path.bezier_curve_to(
                            vec2f(x1 as f32, y1 as f32),
                            vec2f(x2 as f32, y2 as f32),
                            vec2f(x as f32, y as f32),
                        );
                    }
                    PathSegment::QuadTo(Point { x: x1, y: y1 }, Point { x: x2, y: y2 }) => {
                        path.quadratic_curve_to(
                            vec2f(x1 as f32, y1 as f32),
                            vec2f(x2 as f32, y2 as f32),
                        );
                    }
                    PathSegment::Close => {
                        path.close_path();
                    }
                }
            }

            if let Some(ref f) = p.fill() {
                if let usvg::Paint::Color(color) = f.paint() {
                    push_path(
                        scene,
                        &transform,
                        path,
                        &color_u8!(color.red, color.green, color.blue, f.opacity().to_u8()),
                    );
                }
            }

            if let Some(ref s) = p.stroke() {
                if let usvg::Paint::Color(color) = s.paint() {
                    let stroke_style = StrokeStyle {
                        line_width: f32::max(s.width().get(), HAIRLINE_STROKE_WIDTH),
                        line_cap: LineCap::from_usvg_line_cap(s.linecap()),
                        line_join: LineJoin::from_usvg_line_join(
                            s.linejoin(),
                            s.miterlimit().get(),
                        ),
                    };

                    let path = UsvgPathToSegments::new(p.data().segments());
                    let mut outline = Outline::from_segments(path);

                    if let Some(ref dash_array) = s.dasharray() {
                        let dash_array: Vec<f32> = dash_array.iter().map(|&x| x as f32).collect();
                        let mut dash = OutlineDash::new(&outline, &dash_array, s.dashoffset());
                        dash.dash();
                        outline = dash.into_outline();
                    }

                    let mut stroke_to_fill = OutlineStrokeToFill::new(&outline, stroke_style);
                    stroke_to_fill.offset();
                    let outline = stroke_to_fill.into_outline();

                    let mut path = Path2D::new();
                    path.outline = outline;
                    push_path(
                        scene,
                        &transform,
                        path,
                        &color_u8!(color.red, color.green, color.blue, s.opacity().to_u8()),
                    );
                }
            }
        }
        usvg::Node::Group(ref g) => {
            render_nodes(g, scene, global_transform);
        }
        _ => {}
    }
}

#[cfg(feature = "svg")]
pub fn render_nodes(group: &usvg::Group, scene: &mut Scene, global_transform: Transform2F) {
    for child in group.children() {
        render_node(&child, scene, global_transform);
    }
}
