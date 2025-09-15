#version 100

precision highp float;
precision highp sampler2D;

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;

varying vec3 vMaskTexCoord0;
varying vec4 vBaseColor;

float sampleMask(sampler2D maskTexture,
                 vec2 maskTextureSize,
                 vec3 maskTexCoord) {
    vec2 maskTexCoordI = floor(maskTexCoord.xy);
    vec4 texel = texture2D(maskTexture, (floor(maskTexCoordI / vec2(1, 4)) + 0.5) / maskTextureSize);
    float coverage;
    int index = int(mod(maskTexCoordI.y, 4.0));

    if (index == 0) coverage = texel.r;
    else if (index == 1) coverage = texel.g;
    else if (index == 2) coverage = texel.b;
    else coverage = texel.a;

    coverage += maskTexCoord.z;
    coverage = abs(coverage);
    return coverage;
}

void main() {
    // Sample mask.
    float maskAlpha = sampleMask(uMaskTexture0, uMaskTextureSize0, vMaskTexCoord0);

    // Sample color.
    vec4 color = vBaseColor;

    // Apply mask.
    color.a *= maskAlpha;

    // Premultiply alpha.
    color.rgb *= color.a;

	if (floor(mod(gl_FragCoord.x, 16.0)) == 0.0)
		gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
	else if (floor(mod(gl_FragCoord.y, 16.0)) == 0.0)
		gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
	else
		gl_FragColor = color;
}
