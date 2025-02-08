#version 100

// pathfinder/shaders/tile.fs.glsl
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

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;

varying vec3 vMaskTexCoord0;
varying vec4 vBaseColor;

//      Mask UV 0         Mask UV 1
//          +                 +
//          |                 |
//    +-----v-----+     +-----v-----+
//    |           | MIN |           |
//    |  Mask  0  +----->  Mask  1  +------+
//    |           |     |           |      |
//    +-----------+     +-----------+      v       +-------------+
//                                       Apply     |             |       GPU
//                                       Mask +---->  Composite  +---->Blender
//                                         ^       |             |
//    +-----------+     +-----------+      |       +-------------+
//    |           |     |           |      |
//    |  Color 0  +----->  Color 1  +------+
//    |  Filter   |  ×  |           |
//    |           |     |           |
//    +-----^-----+     +-----^-----+
//          |                 |
//          +                 +
//     Color UV 0        Color UV 1

// Masks

float sampleMask(float maskAlpha,
                 sampler2D maskTexture,
                 vec2 maskTextureSize,
                 vec3 maskTexCoord) {
    vec2 maskTexCoordI = floor(maskTexCoord.xy);
    vec4 texel = texture2D(maskTexture, (vec2(floor(maskTexCoordI / vec2(1, 4))) + 0.5) / maskTextureSize);
    float coverage;
    int index = int(mod(maskTexCoordI.y, 4.0));

    if (index == 0) coverage = texel.r;
    else if (index == 1) coverage = texel.g;
    else if (index == 2) coverage = texel.b;
    else coverage = texel.a;

    coverage += maskTexCoord.z;
    coverage = abs(coverage);
    return min(maskAlpha, coverage);
}

// Main function

vec4 calculateColor(sampler2D maskTexture0,
                    vec2 maskTextureSize0,
                    vec3 maskTexCoord0,
                    vec4 baseColor) {
    // Sample mask.
    float maskAlpha = 1.0;
    maskAlpha = sampleMask(maskAlpha, maskTexture0, maskTextureSize0, maskTexCoord0);

    // Sample color.
    vec4 color = baseColor;

    // Apply mask.
    color.a *= maskAlpha;

    // Premultiply alpha.
    color.rgb *= color.a;
    return color;
}

// Entry point
//
// TODO(pcwalton): Generate this dynamically.

void main() {
    // gl_FragColor = vec4(0.01, 0.5, 0.02, 1.0052);
    // gl_FragColor = vec4(vBaseColor.xyz, 1.0052);
    // gl_FragColor = vec4(texture2D(uMaskTexture0, gl_FragCoord.xy).xyz, 1.0);
    gl_FragColor = calculateColor(uMaskTexture0,
                                  uMaskTextureSize0,
                                  vMaskTexCoord0,
                                  vBaseColor);
}
