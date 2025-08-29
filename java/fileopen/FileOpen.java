package fileopen;

import android.net.Uri;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import android.content.ContentResolver;

import TARGET_PACKAGE_NAME.MainActivity;

public class FileOpen {
  public static Uri saved_uri;
  public static MainActivity MainActivity;
  public static ContentResolver resolver;

  public static native void init();

  public static native void finish();

  public static native void saveUri(byte[] data);

  public FileOpen() {}

  public void finishMainActivity() {
	System.exit(0);
  }

  public void OpenFileDialog() {
    MainActivity.OpenFileDialog();
  }

  public void logThis(String text, boolean first) {
	  FileOpen.writeInFile(text, first);
  }

  public static void writeInFile(String text, boolean first) {
    if (saved_uri != null) {
      OutputStream outputStream;
      try {
        String mode = first ? "wt" : "wa";
        outputStream = resolver.openOutputStream(saved_uri, mode);
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(outputStream));
        bw.write(text);
        bw.flush();
        bw.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }
}
