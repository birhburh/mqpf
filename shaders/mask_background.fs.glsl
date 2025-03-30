#version 100
precision highp float;

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;

void main() {
    gl_FragColor = vec4(texture2D(uMaskTexture0, gl_FragCoord.xy / uMaskTextureSize0).xyz, 1.0);
    // gl_FragColor = vec4(0.01, 0.5, 0.02, 1.0052);
}