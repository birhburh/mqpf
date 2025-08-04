//% IMPORTS

import android.app.Activity;
import android.content.SharedPreferences;
import android.net.Uri;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

//% END

//% MAIN_ACTIVITY_BODY

private static final int WRITE_REQUEST_CODE = 101;

public static final String PREFS_NAME = "MyPrefsFile";

public void OpenFileDialog() { 
	SharedPreferences settings = this.getSharedPreferences(PREFS_NAME, 0);
	String urist = settings.getString("saved_uri", null);
	Uri uri = null;
	if (urist != null)
	{
		uri = Uri.parse(urist);
	}
	FileOpen.saved_uri = uri;
	if (FileOpen.saved_uri == null) {
    		Intent myIntent = new Intent(Intent.ACTION_CREATE_DOCUMENT, null);
    		myIntent.setType("text/plain");
    		startActivityForResult(myIntent, WRITE_REQUEST_CODE);
	}
	else
	{
		FileOpen.saveUri(urist.getBytes());
		FileOpen.writeInFile("WOW\n", false);
		// FileOpen.finish();
	}
}
//% END

//% MAIN_ACTIVITY_ON_ACTIVITY_RESULT

if (requestCode == WRITE_REQUEST_CODE) {

	switch (resultCode) {
                case Activity.RESULT_OK:
                    if (data != null
                            && data.getData() != null) {
		Uri uri = data.getData();
		FileOpen.saved_uri = uri;
		FileOpen.saveUri(uri.toString().getBytes());
		getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
		SharedPreferences settings = this.getSharedPreferences(PREFS_NAME, 0);
                SharedPreferences.Editor editor = settings.edit();
                editor.putString("saved_uri", uri.toString());
                editor.apply();

                FileOpen.writeInFile("WRITE DATA\n", false);
                    }
                    break;
                case Activity.RESULT_CANCELED:
                    break;
            }
        FileOpen.writeInFile("FINISH\n", false);
	FileOpen.finish();
}
//% END



//% MAIN_ACTIVITY_ON_CREATE

FileOpen.MainActivity = this;
FileOpen.init();
FileOpen.resolver = getContentResolver();
//% END

