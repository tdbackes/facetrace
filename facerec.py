import cv2,sys,numpy,random,os,math,select,time


def savepeople(continuous, names):
 count=[len(ops) for ops in continuous]
 avcount = reduce(lambda x, y: x + y + 0.0, count) / len(count) + 0.0
 ids=set([i for ops in continuous for i in ops])
 retlist=[]
 for pid in ids:
  count=sum( [1 for ops in continuous if pid in ops ] )
  if count>6:
   retlist.append(names[pid])
 with open('people.txt', 'a+') as f:
  f.write(time.strftime('%X') + ' ' + str(avcount) + ': ' + str(retlist) + '\n')

def prompt() :
 sys.stdout.write('\rCMD: ')
 sys.stdout.flush()

def trainFisher():
 fn_dir='faces'
 
 images = []
 lables = []


 names = {}
 colours={}
 id=0
 for subdir in os.listdir(fn_dir):
  if subdir[0]=='.':
   continue
  names[id]=subdir
  colours[id]=(random.randrange(256),random.randrange(256),random.randrange(256))
  subjectpath=os.path.join(fn_dir, subdir)
  print('subjectpath =  %s'%subjectpath)
  for filename in os.listdir(subjectpath):
   if filename[0]=='.':
    continue
   print('filename =  %s'%filename)
   path=subjectpath +'/'+ filename 
   print('path =  %s'%path)
   lable=id
   images.append(cv2.imread(path,0))
   lables.append(int(lable))
  id+=1

 im_width=images[0].shape[0]
 im_height=images[0].shape[1]


 images = numpy.array(images)
 lables = numpy.array(lables)

 model = cv2.createLBPHFaceRecognizer()
 model.train(images, lables)
 return model, (im_width, im_height), names, colours

def distance(newpoint, oldpoint):
    return math.sqrt((newpoint[0] - oldpoint[0])**2 + (newpoint[1] - oldpoint[1])**2)

def main(model, size, names, colours, webcam):
 im_width, im_height=size
 haar='haar-face.xml'
 downsize = 4
 first = True
 oldpoints = {}
 continuous=[]
 opid = 0
 life = 40
 training={}
 #training[0]=['Noah',20]
 haar_cascade = cv2.CascadeClassifier(haar)

 prompt()
 
 time = 0
 while True:
  continuous.append([oldpoints[op][2][0] for op in oldpoints])
  if len(continuous)>10: continuous.pop(0)
  if not time%10:
   savepeople(continuous,names)
  time+=1
  rval, frame = webcam.read()
  frame=cv2.flip(frame,1,0)
  gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  minigray = cv2.resize(gray, (gray.shape[1]/downsize,gray.shape[0]/downsize))
  faces = haar_cascade.detectMultiScale(minigray)


  faceprediction={}
  facestorage={}
  for f in faces:
   x,y,w,h=[v*downsize for v in f]
   face=gray[y:y+h, x:x+w]
   face_resize=cv2.resize(face,(im_width, im_height))
   prediction=model.predict(face_resize)
   faceprediction[str(f)]=prediction
   facestorage[str(f)]=face_resize




   ########## COMMANDS ###############
   read, write, errors = select.select([sys.stdin], [], [], 0.001)
   for cmd in read:
    cmd = sys.stdin.readline()[:-1]
    cmd=cmd.split(' ')
    if cmd[0]=='train':
     training[int(cmd[1])]=[' '.join(cmd[2:-1]), int(cmd[-1])]
    if cmd[0]=='exit':
     return
    prompt()


   ########## END COMM ###############



  ################################## TIME TRACKING #########################################
  
  todo = [f for f in faces]
  sofar = {}
  for op in oldpoints:
   sofar[op]=''
  extra=[]
  while todo:
   np = todo.pop()
   done = False
   closest = [ (op, distance(np, oldpoints[op][0])) for op in oldpoints]
   closest = [x[0] for x in sorted(closest, key=lambda x: x[1])]
   for cop in closest:
    if sofar[cop]=='' or distance(np, oldpoints[cop][0])<distance(sofar[cop], oldpoints[cop][0]):
     if sofar[cop]!='':
      todo.append(sofar[cop])
     sofar[cop]=np
     done = True
     break
   if not done:
    extra.append(np)
        
  #Update oldpoints
  for op in sofar:
   if sofar[op] !='':
    oldpoints[op]=[sofar[op],life, faceprediction[str(sofar[op])]]
   else:
    oldpoints[op]=[oldpoints[op][0],oldpoints[op][1]-1,oldpoints[op][2]]
      
  #Check for any new points
  for np in extra:
   oldpoints[opid]=[np, life, faceprediction[str(np)]]
   opid+=1
      
  #Check to kill any
  todel=[]
  for op in oldpoints:
   if oldpoints[op][1]==0:
    todel.append(op)
  for op in todel:
   del oldpoints[op]

  ############################## END TIME TRACKING #########################################


  ## TRAINING ##

  traininglist=[index for index in training]
  for index in traininglist:
   if index not in oldpoints or training[index][1]<1:
    print('Deleting %s'%training[index][0])
    model,(w,h), names, colours = trainFisher()
    prompt()
    del training[index]
    continue
   print(training[index][1])
   prompt()
   if oldpoints[index][1]<life:
     continue
   face=facestorage[str(oldpoints[index][0])]
   path='faces/%s/Faces/'%(training[index][0])
   if not os.path.isdir(path):
    os.mkdir(path[:-6])
    os.mkdir(path)
    pin=1
   else:
    pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path) if n[0]!='.' ])[-1] + 1
   path=path + '%s.png'%(pin)
   print('Saving, name=%s, pin=%s'%(training[index][0], pin))
   prompt()
   cv2.imwrite(path, face)
   training[index][1]-=1

  ## NO TRAIN ##


  for op in oldpoints:
   face, age, prediction = oldpoints[op]
   x,y,w,h=[v*downsize for v in face]
   dying = 255//life * (life-age)
   print('Life = %s'%life)
   print('Age = %s'%age)
   cv2.rectangle(frame, (x,y), (x+w, y+h), colours[prediction[0]], 3)
   cv2.putText(frame, '%s: %s %.0f'%(op, names[prediction[0]], 100*(1500-prediction[1])/1500) + '%', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255-dying,255-dying, dying))
   #cv2.putText(frame,'Please enter your PIN', (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255-dying,255-dying, dying))
  cv2.imshow("faces", frame)
  key=cv2.waitKey(1)
  if key==27:
   break

if __name__=='__main__':
 print('Training...')
 model,(w,h), names, colours = trainFisher()
 print('Training finished.')
 webcam = cv2.VideoCapture(0)
 main(model, (w,h), names, colours, webcam)

