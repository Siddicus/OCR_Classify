# from model import my_model
##############################################################################################################################################################################
def cv2contour(args):   
    image = args    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    
    th3=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,5)    
    ret, thresh1 = cv2.threshold(th3,0,255,cv2.THRESH_BINARY_INV )    
    cnts = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)         
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,0), 2)
    for i in range(4):
        cc = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
        ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV)
        c2 = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c2 = c2[0] if len(c2) == 2 else c2[1]        
        for a in c2:
            x,y,w,h = cv2.boundingRect(a)
            if i==3:
                cc.append([x,y,x+w,y+h])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,0), -1)        
    return cc

class test_img_dataset(Dataset):
    def __init__(self,path:str):
        """
        Args:
            path : location of the image folder            
        """
        self.path = path
        self.root = os.listdir(path)       
        
    def __len__(self):
        return len(self.root)
    
    def __getitem__(self,idx):
        image = cv2.imread(os.path.join(self.path,self.root[idx]))
        if image.shape[:2]!= (1024,800):    
            image= cv2.resize(pat,(800,1024))
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        th3=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,4)        
        ret, threshold = cv2.threshold(th3,0,255,cv2.THRESH_OTSU)       
        threshold = threshold[np.newaxis,:]       
        return {"image":torch.tensor(threshold,dtype = torch.float32),"name":self.root[idx]}
    
def inference(test_dataloader,text_on_plot=True):
    output_labels = []
    out_images = []
    for n in test_dataloader:
        sil = os.path.join(test_dataset.path,n["name"][0])    
        u = cv2.imread(sil)
        uy = u.copy()        
        if u.shape[:2]!= (1024,800):    
            u= cv2.resize(u,(800,1024))
            uy = u.copy()
        db_image = u.copy()
        cc= cv2contour(u)    
        out = mode(n["image"]).squeeze().detach()    
        out = F.sigmoid(out).numpy()     
        dop = {0:"text",1:"bar_code",2:"qr_code"}        
        samp = []
        for i in cc:
            ar = []
            ar.append(out[0][i[1]:i[3],i[0]:i[2]].flatten().sum())
            ar.append(out[1][i[1]:i[3],i[0]:i[2]].flatten().sum())
            ar.append(out[2][i[1]:i[3],i[0]:i[2]].flatten().sum())
            if any(ar):            
                if dop[ar.index(np.max(ar))]=="text":
                    ims = db_image[i[1]:i[3],i[0]:i[2]]
                    gray = cv2.cvtColor(ims,cv2.COLOR_BGR2GRAY) 
                    ims = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    ims = cv2.resize(ims,None,fx=1.8,fy=1.8, interpolation=cv2.INTER_CUBIC)
                    ocr = pytesseract.image_to_string(ims)
                    cv2.rectangle(uy,(i[0],i[1]),([i[2],i[3]]),134,thickness = 1)
                    if text_on_plot==True:
                        cv2.putText(uy, f"{dop[ar.index(np.max(ar))]}-{ocr.rstrip()}" , (i[0], i[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 236, 1)
                    else:
                        cv2.putText(uy, f"{dop[ar.index(np.max(ar))]}" , (i[0], i[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 236, 1)              


                    samp.append({"type":dop[ar.index(np.max(ar))],"geometry":[[i[0],i[1]],[i[2],i[3]]],"value":ocr.rstrip()})
                else:
                    cv2.rectangle(uy,(i[0],i[1]),([i[2],i[3]]),134,thickness = 1)
                    cv2.putText(uy, dop[ar.index(np.max(ar))] , (i[0], i[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,236, 1)
                    samp.append({"type":dop[ar.index(np.max(ar))],"geometry":[[i[0],i[1]],[i[2],i[3]]],"value":None})

            else:
                pass
            
        output_labels.append({n["name"][0]:samp})
        out_images.append(uy)
    return output_labels,out_images

class train_unet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model 
        
    def forward(self,x):        
        return self.model(x)
###############################################################################################################################################################################
def main(args):
    # Define your training procedure here

    # script arguments are accessible as follows:
    
    img_path = args.img_path
    #ckpt_path = args.checkpoint_path
    ckpt_path = "https://github.com/Siddicus/OCR_Classify/releases/download/1/cls_res50.ckpt"
    modell = train_unet()
    mode = modell.load_from_checkpoint(checkpoint_path=ckpt_path)
    mode.eval()
    test_dataset =test_img_dataset(path=img_path)
    test_dataloader = DataLoader(test_dataset,batch_size=1)
    localization,image_array=inference(test_dataloader,text_on_plot=True)
    return localization,image_array

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Inference script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('img_path', type=str, help='path to the image')
    parser.add_argument('checkpoint_path', type=str, help='path to your model checkpoint')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    localization,image_array= main(args)
