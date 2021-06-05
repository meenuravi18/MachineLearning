# importing the "tarfile" module
import tarfile
  
# open file
file = tarfile.open('assignment2linux.tar.gz')
  
# extracting file
file.extractall('./Desktop')
  
file.close()