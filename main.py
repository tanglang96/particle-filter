from ParticleFilter import ParticleFilter

def main():
    particles_num = 40
    img_path = r'./test-videos1/Dog1/img'
    out_path = r'./output'
    PF=ParticleFilter(particles_num,img_path,out_path)
    #print(PF.imgs)
    while PF.img_index<len(PF.imgs):
        PF.select()
        PF.propagate()
        PF.observe()
        PF.estimate()

if __name__=='__main__':
    main()