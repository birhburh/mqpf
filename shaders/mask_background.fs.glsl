#version 100
// display fill_mask on the screen for debugging
precision highp float;

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;
uniform vec2 uTileSize;
uniform vec2 uFramebufferSize;

void main() {
	// TODO: Rename to human readable names
	const float scw = 8.0, sch = 2.0;
	float sw = uMaskTextureSize0.x, scpw = sw / scw;

	const float scrolly=0.0;
	const float dcw = 160.0, dch = 160.0;
	float dcpw = 16.0;
	float dw = dcpw * dcw;
	float dx = gl_FragCoord.x;
	float dy = gl_FragCoord.y;
    if (floor(mod(dx, dcw)) == 0.0)
        gl_FragColor = vec4(0.1,0.5,0.2,1.0);
    else if (floor(mod(dy, dch)) == 0.0)
        gl_FragColor = vec4(0.1,0.5,0.2,1.0);
    else if (dx <= dw) {
		float dcpx = dx/dcw;
		float dcpy = dy/dch+scrolly;
		float doffx=mod(dx,dcw);
		float doffy=mod(dy,dch);
		float scpx=mod(dcpx+dcpw*dcpy, scpw);
		float scpy=(dcpx+dcpw*dcpy)/scpw;
		float soffx=doffx*(scw/dcw);
		float soffy=doffy*(sch/dch);
		float sx=scpx*scw+soffx;
		float sy=scpy*sch+soffy;
        gl_FragColor = vec4(texture2D(uMaskTexture0, vec2(sx, sy) / uMaskTextureSize0).xyz, 1.0);
    }
    // gl_FragColor = vec4(texture2D(uMaskTexture0, gl_FragCoord.xy / uMaskTextureSize0 / vec2(4.0, 4.0)).xyz, 1.0);
    // gl_FragColor = vec4(0.01, 0.5, 0.02, 1.0052);
}
