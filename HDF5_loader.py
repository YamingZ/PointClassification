import h5py
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class HDF5Data(object):
    def __init__(self,filepath):
        self.h5file = h5py.File(filepath)
    def getKeys(self):
        return [key for key in self.h5file.keys()]
    def getValue(self,key):
        return self.h5file[key][:]
    def showInFO(self):
        for key in self.getKeys():
            print(key)
            print(self.getValue(key).shape)

class Model(object):
    def __init__(self,pointsize=2,color=None,vertex=None,lines=None,visible=True,colorful=False):
        self.visible = visible
        self.vertices = vertex
        self.lines = lines
        self.color = color
        self.ptsize = pointsize
        self.colorful = colorful
    def draw(self):
        if self.visible:
            if self.lines != None:
                glEnable(GL_BLEND)
                glEnable(GL_LINE_SMOOTH)
                for line in self.lines:
                    glColor3f(1-line[6], 1-line[6], 1-line[6])
                    glLineWidth(line[6]*3)
                    glBegin(GL_LINES)
                    glVertex3f(line[0], line[1], line[2])
                    glVertex3f(line[3], line[4], line[5])
                    glEnd()
            else:
                glPointSize(self.ptsize)
                glBegin(GL_POINTS)
                i=0
                for point in self.vertices:
                    if self.colorful:
                        color = self.color[i]
                    else:
                        color = self.color
                    glColor3f(color[0],color[1],color[2])
                    glVertex3f(point[0],point[1],point[2])
                    i+=1
                glEnd()

    def SetVisible(self):
        self.visible = ~self.visible

class Camera(object):
    THETA_MIN = 0.1
    THETA_MAX = 179.9
    R_MIN = 0.02
    R_MAX = 40
    PI_180 = 0.017453293
    def __init__(self):
        self._lastPos= (0,0)
        self._centerPos= (0,0,0)
        self._R=4.0
        self._THETA_DEG = 60
        self._PHI_DEG = 0
        self.left_Btn_press = False
    def getCameraPos(self):
        phi = self.PI_180 * self._PHI_DEG
        theta = self.PI_180 * self._THETA_DEG
        eye_x = self._R * math.sin(theta) * math.cos(phi)
        eye_z = self._R * math.sin(theta) * math.sin(phi)
        eye_y = self._R * math.cos(theta)
        return eye_x,eye_y,eye_z
    def getCenterPos(self):
        return self._centerPos

    def mouseMove(self,x,y):
        if self.left_Btn_press:
            dx = x - self._lastPos[0]
            dy = y - self._lastPos[1]
            self._lastPos = (x, y)
            self._PHI_DEG += dx/2
            self._THETA_DEG -= dy/2
            if self._THETA_DEG < self.THETA_MIN:
                self._THETA_DEG = self.THETA_MIN
            elif self._THETA_DEG > self.THETA_MAX:
                self._THETA_DEG = self.THETA_MAX

    def mousePress(self,button,state,x,y):
        if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
            self._lastPos = (x, y)
            self.left_Btn_press = True
        if button == 3 or button == 4:  #mouse wheel
            dir = 1 if(button == 3)else -1
            self._R +=  dir / 10.0
            if self._R < self.R_MIN:
                self._R = self.R_MIN
            elif self._R > self.R_MAX:
                self._R = self.R_MAX

    def showCameraInfo(self):
        print(self._R,self._THETA_DEG,self._PHI_DEG,self.getCameraPos())

class Axis(object):
    def __init__(self,size,step):
        self.visible = True
        self.size = size
        self.vertices = []
        for i in range(int(size[0]/step)+1):  #x
            line = [-size[0]/2.0 + step*i,0.0,-size[1]/2.0,-size[0]/2.0 + step*i,0.0, size[1]/2.0]
            self.vertices.append(line)
        for j in range(int(size[1]/step)+1): #z
            line = [-size[0]/2.0,0.0,-size[1]/2.0+step*j,size[0]/2.0,0.0, -size[1]/2.0+step*j]
            self.vertices.append(line)
        for i in range(int(size[0]/step)+1):  #x
            line = [-size[0]/2.0 + step*i,-size[1]/2.0,0.0,-size[0]/2.0 + step*i,size[1]/2.0,0.0]
            self.vertices.append(line)
        for j in range(int(size[1]/step)+1): #y
            line = [-size[0]/2.0,-size[1]/2.0+step*j,0.0,size[0]/2.0, -size[1]/2.0+step*j,0.0]
            self.vertices.append(line)

    def SetVisible(self):
        self.visible = not self.visible
        print(self.visible)

    def draw(self):
        if self.visible:
            glLineWidth(1.0)
            glBegin(GL_LINES)
            glColor3f(1.0, 0.0, 0.0)    #x
            glVertex3f(-self.size[0]/2, 0, 0)
            glVertex3f(self.size[0]/2, 0, 0)
            glColor3f(0.0, 0.0, 1.0)    #z
            glVertex3f(0, 0, -self.size[1]/2)
            glVertex3f(0, 0, self.size[1]/2)
            glColor3f(0.0, 1.0, 0.0)    #y
            glVertex3f(0, -2.5, 0)
            glVertex3f(0, 2.5, 0)
            for line in self.vertices:
                glColor3f(1.0,1.0,1.0)
                glVertex3f(line[0],line[1],line[2])
                glVertex3f(line[3],line[4],line[5])
            glEnd()
            glPushMatrix()
            glTranslated(0.0, 0.0, 2.5)
            glutWireCone(0.04, 0.3, 8, 8)
            glTranslated(2.5, 0.0,-2.5)
            glRotated(90.0, 0, 1.0, 0)
            glutWireCone(0.04, 0.3, 8, 8)
            glTranslated(0.0, 2.5, -2.5)
            glRotated(-90.0, 1.0, 0.0, 0.0)
            glutWireCone(0.04, 0.25, 8, 8)
            glPopMatrix()

class View3D(object):
    def __init__(self,models,camera,background,size=(1024,640),name="view"):
        self.models = models
        self.camera = camera
        self.axis = Axis((5,5),0.25)
        self.model_ID = 0
        self.model_show_list = []
        self.background = background
        glutInit()
        glutInitWindowSize(size[0],size[1])
        glutCreateWindow(name)
        self.createGLUTMenus()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        self.OpenGL_init()
        glutReshapeFunc(self.reshape)
        glutDisplayFunc(self.display)
        glutIdleFunc(self.display)
        glutMouseFunc(self.mousemove)
        glutMotionFunc(self.mousemotion)
        glutMainLoop()

    def OpenGL_init(self):
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glClearColor(self.background[0],self.background[1],self.background[2],self.background[3])   #0.5,0.6,0.5

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        ca_x, ca_y, ca_z = self.camera.getCameraPos()
        ce_x, ce_y, ce_z = self.camera.getCenterPos()
        gluLookAt(ca_x,ca_y,ca_z,ce_x,ce_y,ce_z,0.0,1.0,0.0)
        for i in self.model_show_list:
            self.models[i].draw()
        self.models[self.model_ID].draw()
        self.axis.draw()
        glutSwapBuffers()

    def createGLUTMenus(self):
        glutCreateMenu(self.processMenuEvents)
        glutAddMenuEntry("Next Model", 1)
        glutAddMenuEntry("Last  Model", 2)
        glutAddMenuEntry("Hold Model", 3)
        glutAddMenuEntry("Show Axis", 4)
        glutAttachMenu(GLUT_RIGHT_BUTTON)
        # return 0

    def processMenuEvents(self,option):
        if option == 1:
            if self.model_ID < len(self.models)-1:
                self.model_ID += 1
        elif option == 2:
            if self.model_ID > 0:
                self.model_ID -= 1
        elif option == 3:
            if self.model_ID in self.model_show_list:
                self.model_show_list.remove(self.model_ID)
            else:
                self.model_show_list.append(self.model_ID)
        elif option == 4:
            self.axis.SetVisible()
        else:
            pass
        return 0

    def reshape(self,w,h):
        glViewport(0, 0, w, h) # 改变显示区域，起始位置为客户端窗口左下角（非坐标原点）
        glMatrixMode(GL_PROJECTION) # 修改投影矩阵
        glLoadIdentity() # 导入单位阵
        gluPerspective(60.0, w / h, 0.01, 50.0) # 宽高比改为当前值，视线区域与屏幕大小一致；
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def mousemove(self,button,state,x,y):
        self.camera.mousePress(button,state,x,y)

    def mousemotion(self,x,y):
        self.camera.mouseMove(x,y)


if __name__ == '__main__':
    h5 = HDF5Data('/home/ym/PycharmProjects/Fundamentals_of_Python/PointClassfication/data/modelnet/modelnet40_ply_hdf5_2048/ply_data_train4.h5')
    h5.showInFO()

    coor = np.array(h5.getValue('data'))[4, :, :]
    coor = np.expand_dims(coor, axis=0)
    # coor = rotate_point_cloud(coor)
    # coor = normalized_point_cloud(coor)
    line = [[0.0,0.0,0.0,-1.0,1.0,1.0,1.5],[0.0,0.0,0.0,2.0,2.0,2.0,3.6]]
    model = Model([1.0,1.0,0.0],lines=line)
    camera = Camera()
    view = View3D([model],camera,[0.5,0.6,0.5,0.0],(1024,640),"3Dview")

