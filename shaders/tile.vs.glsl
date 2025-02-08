#version 100

// pathfinder/shaders/tile.vs.glsl
//
// Copyright © 2020 The Pathfinder Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

precision highp float;

#ifdef GL_ES
precision highp sampler2D;
#endif

uniform sampler2D uTextureMetadata;

uniform mat4 uTransform;
uniform vec2 uTileSize;
uniform ivec2 uTextureMetadataSize;

attribute vec2 aTileOffset;
attribute vec2 aTileOrigin;
attribute vec4 aMaskTexCoord0;
attribute float aColor;
attribute vec2 aCtrlBackdrop;

varying vec3 vMaskTexCoord0;
varying vec4 vBaseColor;

vec4 fetchUnscaled(sampler2D srcTexture, vec2 scale, vec2 originCoord, float entry) {
    return texture2D(srcTexture, (originCoord + vec2(0.5) + vec2(entry, 0)) * scale);
}

void main() {
    vec2 tileOrigin = vec2(aTileOrigin), tileOffset = vec2(aTileOffset);
    vec2 position = (tileOrigin + tileOffset) * uTileSize;

    vec2 maskTileCoord = vec2(aMaskTexCoord0.x, aMaskTexCoord0.y + 256.0 * aMaskTexCoord0.z);
    vec2 maskTexCoord0 = (vec2(maskTileCoord) + tileOffset) * uTileSize;
    if (aCtrlBackdrop.y == 0.0 && aMaskTexCoord0.w != 0.0) {
        gl_Position = vec4(0.0);
        return;
    }

    vec2 metadataScale = vec2(1.0) / vec2(uTextureMetadataSize);
    vec2 metadataEntryCoord = vec2(mod(aColor, 128.0), aColor / 128.0);
    vec4 baseColor       = fetchUnscaled(uTextureMetadata, metadataScale, metadataEntryCoord, 0.0);
    vBaseColor = baseColor;

    vMaskTexCoord0 = vec3(maskTexCoord0, float(aCtrlBackdrop.y));
    gl_Position = uTransform * vec4(position, 0.0, 1.0);
}
