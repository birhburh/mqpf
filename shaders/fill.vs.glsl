#version 100

// pathfinder/shaders/fill.vs.glsl
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

precision highp float;
precision highp sampler2D;

uniform vec2 uFramebufferSize;
uniform vec2 uTileSize;

attribute vec2 aTessCoord;
attribute vec4 aLineSegment;
attribute float aTileIndex;

varying vec2 vFrom;
varying vec2 vTo;

vec2 computeTileOffset(float tileIndex, float stencilTextureWidth, vec2 tileSize) {
    float tilesPerRow = float(stencilTextureWidth / tileSize.x);
    vec2 tileOffset = vec2(mod(tileIndex, tilesPerRow), floor(tileIndex / tilesPerRow));
    return vec2(tileOffset) * tileSize * vec2(1.0, 0.25);
}

vec4 computeVertexPosition(float tileIndex,
                           vec2 tessCoord,
                           vec4 packedLineSegment,
                           vec2 tileSize,
                           vec2 framebufferSize,
                           out vec2 outFrom,
                           out vec2 outTo) {
    vec2 tileOrigin = computeTileOffset(float(tileIndex), framebufferSize.x, tileSize);

    vec4 lineSegment = vec4(packedLineSegment) / 256.0;
    vec2 from = lineSegment.xy, to = lineSegment.zw;

    vec2 position;
    if (tessCoord.x == 0.0)
        position.x = floor(min(from.x, to.x));
    else
        position.x = ceil(max(from.x, to.x));
    if (tessCoord.y == 0.0)
        position.y = floor(min(from.y, to.y));
    else
        position.y = tileSize.y;
    position.y = floor(position.y * 0.25);

    // Since each fragment corresponds to 4 pixels on a scanline, the varying interpolation will
    // land the fragment halfway between the four-pixel strip, at pixel offset 2.0. But we want to
    // do our coverage calculation on the center of the first pixel in the strip instead, at pixel
    // offset 0.5. This adjustment of 1.5 accomplishes that.
    vec2 offset = vec2(0.0, 1.5) - position * vec2(1.0, 4.0);
    outFrom = from + offset;
    outTo = to + offset;

    vec2 globalPosition = (tileOrigin + position) / framebufferSize * 2.0 - 1.0;

    return vec4(globalPosition, 0.0, 1.0);
}

void main() {
    // vec2 aTessCoord = aTessCoord;
    // vec4 aLineSegment = aLineSegment;
    // float aTileIndex = aTileIndex;
    // gl_Position = vec4(aTessCoord, 0.0, 1.0);
    // computeVertexPosition(float(aTileIndex),
    //                             aTessCoord,
    //                             aLineSegment,
    //                             uTileSize,
    //                             uFramebufferSize,
    //                             vFrom,
    //                             vTo);
    gl_Position = computeVertexPosition(float(aTileIndex),
                                        aTessCoord,
                                        aLineSegment,
                                        uTileSize,
                                        uFramebufferSize,
                                        vFrom,
                                        vTo);
}
