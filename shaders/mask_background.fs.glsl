#version 100
// display fill_mask on the screen for debugging
precision highp float;

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;
uniform vec2 uTileSize;
uniform vec2 uFramebufferSize;

void main() {
	// TODO: Rename to human readable names
	const float scw = 16.0, sch = 4.0;
	float sw = uMaskTextureSize0.x, scpw = sw / scw;

	// const float scrolly=19.0;
	const float scrolly=0.0;
	// const float dcw = 400.0, dch = 100.0;
	const float dcw = 160.0, dch = 40.0;
	// dcpw - dest cell pos w - width of dest in cells
	float dcpw = 5.0;
	float dx = gl_FragCoord.x;
	float dy = gl_FragCoord.y;
    if (floor(mod(dx, dcw)) == 0.0)
        gl_FragColor = vec4(0.1,0.5,0.2,1.0);
    else if (floor(mod(dy, dch)) == 0.0)
        gl_FragColor = vec4(0.1,0.5,0.2,1.0);
    else if (dx <= dcpw*dcw) {
		// dcpx - dest cell pos x
		float dcpx = floor(dx/dcw);
		// dcpy - dest cell pos y
		float dcpy = floor(dy/dch)+scrolly;
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
}
