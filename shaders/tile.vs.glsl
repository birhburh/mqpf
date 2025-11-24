#version 100

precision highp float;
precision highp sampler2D;

uniform sampler2D uTextureMetadata;

uniform mat4 uTransform;
uniform vec2 uTileSize;
uniform float uTileZoom;
uniform ivec2 uTextureMetadataSize;

attribute vec2 aTileOffset;
attribute vec2 aTileOrigin;
attribute float aMaskTexCoord;
attribute float aColor;
attribute float aCtrlBackdrop;

varying vec3 vMaskTexCoord0;
varying vec4 vBaseColor;

void main() {
    vec2 position = (aTileOrigin + aTileOffset) * uTileSize.x * uTileZoom - uTileSize.x * uTileZoom * 10.0;

    // depacking 2 ints from aMaskTexCoord
    int aMaskTexCoord_y = int(mod(aMaskTexCoord / 256.0, 256.0));
    int aMaskTexCoord_x = int(mod(aMaskTexCoord, 256.0));

    vec2 maskTileCoord = vec2(aMaskTexCoord_x, aMaskTexCoord_y);
    vec2 maskTexCoord0 = (maskTileCoord + aTileOffset) * uTileSize;

    // aMaskTexCoord != INVALID
    if (aCtrlBackdrop == 0.0 && abs(aMaskTexCoord - 65536.0) < 0.00001) {
        gl_Position = vec4(0.0);
        return;
    }

    vec2 metadataScale = vec2(1.0) / vec2(uTextureMetadataSize);
    vec2 metadataEntryCoord = vec2(mod(aColor, 128.0), aColor / 128.0);
    vBaseColor = texture2D(uTextureMetadata, (metadataEntryCoord + vec2(0.5)) * metadataScale);

    vMaskTexCoord0 = vec3(maskTexCoord0, aCtrlBackdrop);
    gl_Position = uTransform * vec4(position, 0.0, 1.0);
}
