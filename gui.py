import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from AudioDelayFixer import AudioDelayFixer
import timeit 


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Video Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)
    
    def setText(self , text):
        super().setText(text)
    
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)
        mainLayout = QVBoxLayout()
        self.file_path = ""

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            self.file_path = event.mimeData().urls()[0].toLocalFile()
            self.photoViewer.setText('\n\n Video is processing \n\n')
            start = timeit.default_timer()
            adf = AudioDelayFixer(demo.getFilePath())
            adf.fixAudioDelay()
            stop = timeit.default_timer()
            print('Time: ', stop - start)
            self.photoViewer.setText('\n\n Audio delay is fixed within {time:.2f} seconds \n\n'.format(time = stop - start))

            event.accept()
        else:
            event.ignore()
    
    def getFilePath(self):
        return self.file_path
    
app = QApplication(sys.argv)
demo = AppDemo()
demo.show()     

sys.exit(app.exec_())
