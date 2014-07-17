package classifier

import java.io.File

object Tempfile {
  def apply(directoryPath: String) = {
    val file = File.createTempFile("classifier", "", new File(directoryPath))
    file.deleteOnExit()
    file
  }
}
