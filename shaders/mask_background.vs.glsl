#version 100
attribute vec2 in_pos;

uniform vec2 uMaskTextureSize0;

void main() {
    gl_Position = vec4(in_pos * 2.0 - 1.0, 0, 1);
    // gl_Position = vec4(in_pos, 0, 1);
}