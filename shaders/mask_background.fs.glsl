#version 100
// display fill_mask on the screen for debugging
precision highp float;

uniform sampler2D uMaskTexture0;
uniform vec2 uMaskTextureSize0;
uniform vec2 uTileSize;
uniform vec2 uFramebufferSize;

void main() {
	// TODO: Rename to human readable names
	const float scomp = 4.0; // mask pixels stored in single pixel
	const float scw = 16.0, sch = 4.0;
	float sw = uMaskTextureSize0.x, scpw = sw / scw;

	// const float scrolly=19.0;
	const float scrolly=0.0;
	// const float dcw = 400.0, dch = 100.0;
	const float zoom = 6.0;
	const float dcw = scw * zoom, dch = sch*zoom*scomp;
	// dcpw - dest cell pos w - width of dest in cells
	float dcpw = 5.0;
	float dx = gl_FragCoord.x;
	float dy = gl_FragCoord.y;
    if (dx <= dcpw*dcw+1.0) {
		if (floor(mod(dx, dcw)) == 0.0)
			gl_FragColor = vec4(0.1,0.5,0.2,1.0);
		else if (floor(mod(dy, dch)) == 0.0)
        	gl_FragColor = vec4(0.1,0.5,0.2,1.0);
		else {
			// dcpx - dest cell pos x
			float dcpx = floor(dx/dcw);
			// dcpy - dest cell pos y
			float dcpy = floor(dy/dch)+scrolly;
			float doffx=mod(dx,dcw);
			float doffy=mod(dy,dch);
			float scpx=mod(dcpx+dcpw*dcpy, scpw);
			float scpy=floor((dcpx+dcpw*dcpy)/scpw);
			float soffx=doffx*(scw/dcw);
			float soffy=doffy*(sch/dch);
			float sx=scpx*scw+soffx;
			float sy=scpy*sch+soffy;

        	vec4 texel = texture2D(uMaskTexture0, vec2(sx, sy) / uMaskTextureSize0);
			gl_FragColor = vec4(texel.xyz, 1.0);
			// return;
			float coverage;

			// decompressing mask
			float poffy = doffy-(dch/sch*floor(soffy)); // y offset inside one dest scaled pixel
			int index = int(floor(poffy/((dch/sch))*scomp));
			// gl_FragColor = vec4(float(index)/scomp, 0.0, 0.0, 1.0);
			// return;
			if (index == 0) coverage = texel.r;
			else if (index == 1) coverage = texel.g;
			else if (index == 2) coverage = texel.b;
			else coverage = texel.a;

			coverage = abs(coverage);
			gl_FragColor = vec4(coverage, coverage, coverage, 1.0);
		}
	}
}
