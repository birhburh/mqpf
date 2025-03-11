use {
    macroquad::{
        miniquad::{
            conf::{AppleGfxApi, Platform},
            window::screen_size,
        },
        prelude::*,
    }, mqpf::{
        push_path, ArcDirection, Contour, Outline, Path2D, PushSegmentFlags, Renderer,
        Scene, Segment, SegmentFlags,
    }, pathfinder_geometry::{
        line_segment::LineSegment2F, rect::RectF, transform2d::{Matrix2x2F, Transform2F}, vector::{vec2f, Vector2F}
    }, std::mem, usvg::{
        tiny_skia_path::{PathSegment, Point},
        LineCap as UsvgLineCap, LineJoin as UsvgLineJoin, Tree as SvgTree,
    }
};

fn window_conf() -> Conf {
    let apple_gfx_api = AppleGfxApi::OpenGl;
    let high_dpi = true;
    let window_width = 600;
    let window_height = window_width * 3 / 4;
    Conf {
        window_title: format!("Tiger").to_owned(),
        platform: Platform {
            apple_gfx_api,
            // blocking_event_loop: true,
            ..Default::default()
        },
        // fullscreen: true,
        window_width,
        window_height,
        high_dpi,
        ..Default::default()
    }
}

const HAIRLINE_STROKE_WIDTH: f32 = 0.0333;
const EPSILON: f32 = 0.0001;
const TOLERANCE: f32 = 0.01;

struct UsvgPathToSegments<I>
where
    I: Iterator<Item = PathSegment>,
{
    iter: I,
    first_subpath_point: Vector2F,
    last_subpath_point: Vector2F,
    just_moved: bool,
}

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

struct DashState<'a> {
    output: Contour,
    dashes: &'a [f32],
    current_dash_index: usize,
    distance_left: f32,
}

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

struct ContourDash<'a, 'b, 'c> {
    input: &'a Contour,
    output: &'b mut Outline,
    state: &'c mut DashState<'a>,
}

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

/// Transforms a stroke into a dashed stroke.
pub struct OutlineDash<'a> {
    input: &'a Outline,
    output: Outline,
    state: DashState<'a>,
}

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

struct ContourStrokeToFill<'a> {
    input: &'a Contour,
    output: Contour,
    radius: f32,
    join: LineJoin,
}

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

pub struct OutlineStrokeToFill<'a> {
    input: &'a Outline,
    output: Outline,
    style: StrokeStyle,
}

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

fn load_scene() -> SvgTree {
    let svg_data = include_bytes!("../svgs/Ghostscript_Tiger.svg");
    SvgTree::from_data(svg_data, &usvg::Options::default()).unwrap()
}

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

fn render_nodes(group: &usvg::Group, scene: &mut Scene, global_transform: Transform2F) {
    for child in group.children() {
        render_node(&child, scene, global_transform);
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut framebuffer_size = screen_size();

    let context = unsafe { get_internal_gl().quad_context };
    let mut renderer = Renderer::new(context, framebuffer_size, color_u8!(77, 77, 82, 255));

    let mut saved_width = 0.0;
    let mut saved_height = 0.0;

    let tree = load_scene();

    let mut canvas_scene = Scene {
        view_box: RectF::new(vec2f(0.0, 0.0), vec2f(framebuffer_size.0, framebuffer_size.1)),
        ..Default::default()
    };

    let start_time = get_time();
    loop {
        clear_background(DARKGRAY);

        if screen_width() != saved_width || screen_height() != saved_height {
            saved_width = screen_width();
            saved_height = screen_height();

            framebuffer_size = screen_size();
            renderer.update_viewport(framebuffer_size);

            canvas_scene = Scene {
                view_box: RectF::new(vec2f(0.0, 0.0), vec2f(framebuffer_size.0, framebuffer_size.1)),
                ..Default::default()
            };

            let side_size = tree.size().width().min(tree.size().height());
            let scale = if screen_width() < screen_height() {
                framebuffer_size.0 / side_size * 0.9
            } else {
                framebuffer_size.1 / side_size * 0.9
            };

            let mut transform = Transform2F::from_translation(vec2f(
                framebuffer_size.0 / 2.0 - side_size * scale / 2.0,
                framebuffer_size.1 / 2.0 - side_size * scale / 2.0,
            ));
            transform *= Transform2F::from_scale(vec2f(scale, scale));

            render_nodes(&tree.root(), &mut canvas_scene, transform);
        }

        renderer.render(&canvas_scene);

        if get_time() - start_time > 10.0 {
            break;
        }
        next_frame().await;
    }
}
