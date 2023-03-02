import os
import re
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import math

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.Jpg','.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    """Return True if the file is an image.

    >>> is_image_file('front_1.jpg')
    True
    >>> is_image_file('bs')
    False
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """Return a list that constains all the image paths in the given dir.

    >>> dir = '/home/jaren/data/train'
    >>> len(make_dataset(dir))
    3566
    >>> dir = '/home/jaren/data/test'
    >>> len(make_dataset(dir))
    2027
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' %dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                id = get_id(path,fname)
                pose = get_pose(path,fname)
                if pose != 9999:
                    images.append({'path': path,
                                    'id': id,
                                    'pose': pose,
                                    'name': fname})

    return images

def split_with_same_id(samples):
    """
    split the list samples to sublists that with the same id.
    """
    result = []
    if len(samples)==0:
        return result
    temp0=[]
    count=False
    result.append([samples[0]])
    if samples[0]['pose']==0:
        count=True
    for i in range(1, len(samples)):
        if samples[i-1]['id']==samples[i]['id'] and len(result[-1])<200:
            result[-1].append(samples[i])
            if samples[i]['pose']==0:
                temp0=samples[i]
                count=True

        else:
            if len(temp0)!=0 and temp0['id']==samples[i]['id'] and len(result[-1])>=200:
                result.append([temp0])
                result[-1].append(samples[i])
                if samples[i]['pose']==0:
                    count=False
            else:
                if count==False:
                    result.pop()
                result.append([samples[i]])
                if samples[i]['pose']==0:
                    count=True
                else:
                    count=False

    return result


def default_loader(path):
    return Image.open(path).convert('RGB')

def get_id(path,fname):
    """Return the id of the image.

    >>> path = '/home/jaren/data/train//001/front_1.jpg'
    >>> get_id(path)
    1
    >>> path = '/home/jaren/data/train//034/front_1.jpg'
    >>> get_id(path)
    34
    """
    '''p = re.compile(r'\d{2}')
    
    k= re.findall(p, path)

    id=int(k[0]);
    '''
    id=''
    for i in range(len(fname)):
        if fname[i] == '_':
            for j in range(i+1,len(fname)):
                if fname[j]=='_':
                    break
                else:
                    id=id+fname[j]
            break


    
    return int(id);






def get_pose(path,fname):
    """Return the pose of the image.
    profile->False
    Frontal->True

    >>> path = '/home/jaren/data/train//001/front_1.jpg'
    >>> get_pose(path)
    True
    >>> path = '/home/jaren/data/train//034/profile_1.jpg'
    >>> get_pose(path)
    False
    """
    ''' q = re.compile(r'[_]\d{2}[_][0]')
    
    if  re.search(q, path):
        k= re.findall(q, path)
        #print(k)
        st=str(k)
        pose=int(st[6])
        #print("pose= ",pose)
        if pose==0:
            return 0
    
    p = re.compile(r'\d{2}')
    k= re.findall(p, path)
    #print(k)
    pose=int(k[2]);
    
    """result = True if re.search(p, path) else False"""
   
    if pose==45:
        return 1
    
    return 2'''

    pose=''
    count=0
    for i in range(len(fname)):
        if fname[i] == '_':
            count+=1
        if(count<2):
            continue
        
        for j in range(i+1,len(fname)):
            if fname[j]=='.':
                break
            else:
                pose=pose+fname[j]
        break
    if(int(pose)>=-3 and int(pose)<=3):
        return 0
    elif (int(pose)>3 and int(pose)<=12):
        return 1
    elif (int(pose)>12 and int(pose)<=21):
        return 2
    elif (int(pose)>21 and int(pose)<=30):
        return 3
    elif (int(pose)>30 and int(pose)<=40):
        return 4
    elif (int(pose)>40 and int(pose)<=50):
        return 5
    elif (int(pose)>50 and int(pose)<=90):
        return 6
    elif (int(pose)<-3 and int(pose)>=-12):
        return 7
    elif (int(pose)<-12 and int(pose)>=-21):
        return 8
    elif (int(pose)<-21 and int(pose)>=-30):
        return 9
    elif (int(pose)<-30 and int(pose)>=-40):
        return 10
    elif (int(pose)<-40 and int(pose)>=-50):
        return 11
    elif (int(pose)<-50 and int(pose)>=-90):
        return 12
    else:
        return 9999
        

    
   


def show_sample(sample):
    """
    Plot the Tensor sample.
    input: The dict sample of the dataset.
    """
    image = []
    pose = []
    identity = []
    name = []
    for i in range(len(sample)):
        image.append(sample[i]['image'])
        pose.append(sample[i]['pose'])
        identity.append(sample[i]['identity'])
        name.append(sample[i]['name'])
    image = [item for sublist in image for item in sublist]
    pose = [item for sublist in pose for item in sublist]
    identity = [item for sublist in identity for item in sublist]
    name = [item for sublist in name for item in sublist]
    for j in range(len(identity)):

        img = 0.5*image[j] + 0.5
        img = transforms.ToPILImage()(img)

        fig = plt.figure(1)
        ax = fig.add_subplot(4, math.ceil(len(pose)/4), j+1)

        ax.set_title('Frontal:{0}\nIdentity:{1}\nName:{2}'.format(pose[j], identity[j], name[j]))
        plt.imshow(img)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()

if __name__ == '__main__':
    dir = '/home/jaren/data//train'
    samples = make_dataset(dir)
    ids = split_with_same_id(samples)
