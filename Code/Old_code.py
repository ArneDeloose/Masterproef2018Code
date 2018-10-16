def spect_loop2(file_name): #full plot, extract 100 ms every time
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern, _, _,_,_=AD.set_parameters()
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    full_plot=AD.spect_plot(samples,sample_rate)
    for i in range(steps):
        for j in range(10):
            spect_norm=full_plot[:, int(i*1710+171*j):int(i*1710+171*(j+1))]
            ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
            if dummy_flag:
                rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
            spectros[i*10+j]=spect_norm
    for j in range(microsteps):
        spect_norm=full_plot[:, int((i+1)*1710+171*j):int((i+1)*1710+171*(j+1))]
        ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
        if dummy_flag:
            rectangles[(i+1)*10+j], regions[(i+1)*10+j]=AD.ROI2(ctrs, spect_norm)
        spectros[(i+1)*10+j]=spect_norm
    rectangles2, regions2=AD.overload(rectangles, regions)
    return(rectangles2, regions2, spectros)

def spect_loop3(file_name): #one plot for 100 ms
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern, _, _,_,_=AD.set_parameters()
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    for i in range(steps):
        for j in range(10):
            samples_dummy=samples[int(i*sample_rate+sample_rate*j/10):int(i*sample_rate+sample_rate*(j+1)/10)]
            spect_norm=AD.spect_plot(samples_dummy,sample_rate)
            ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
            if dummy_flag:
                rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
            spectros[i*10+j]=spect_norm
    for j in range(microsteps):
        samples_dummy=samples[int((i+1)*sample_rate+sample_rate*j/10):int((i+1)*sample_rate+sample_rate*(j+1)/10)]
        spect_norm=AD.spect_plot(samples_dummy,sample_rate)
        ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
        if dummy_flag:
            rectangles[(i+1)*10+j], regions[(i+1)*10+j]=AD.ROI2(ctrs, spect_norm)
        spectros[(i+1)*10+j]=spect_norm
    rectangles2, regions2=AD.overload(rectangles, regions)
    return(rectangles2, regions2, spectros)

def spect_loop4(file_name): #hybrid code, full plot for 1s, extract 100 ms every time
    #Function creates a dictionary 'rectangles' containing coordinates of the ROIs per image
    #Each image is 100 ms, number within dictionary indicates 
    #image number (e.g rectangles(45: ...) is 4500ms to 4600ms or 4.5 secs to 4.6 secs)
    #Empty images are skipped
    X, kern, _, _,_,_=AD.set_parameters()
    sample_rate, samples, t, total_time,steps, microsteps= AD.spect(file_name);
    rectangles={};
    regions={};
    spectros={};
    for i in range(steps):
        samples_dummy=samples[int(i*sample_rate):int((i+1)*sample_rate)]
        full_plot=AD.spect_plot(samples_dummy,sample_rate)
        for j in range(10):
            spect_norm=full_plot[:, int(171*j):int(171*(j+1))]
            ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
            if dummy_flag:
                rectangles[i*10+j], regions[i*10+j]=AD.ROI2(ctrs, spect_norm)
            spectros[i*10+j]=spect_norm
    samples_dummy=samples[int((i+1)*sample_rate):] #until the end
    full_plot=AD.spect_plot(samples_dummy,sample_rate)
    for j in range(microsteps):
        spect_norm=full_plot[:, int(171*j):int(171*(j+1))]
        ctrs, dummy_flag=AD.ROI(spect_norm, kern, X)
        if dummy_flag:
            rectangles[(i+1)*10+j], regions[(i+1)*10+j]=AD.ROI2(ctrs, spect_norm)
        spectros[(i+1)*10+j]=spect_norm
    rectangles2, regions2=AD.overload(rectangles, regions)
    return(rectangles2, regions2, spectros)    

def create_template_set_old(): #temp function storing a template set
    file_name1='ppip-1µl1µA044_AAT.wav' #ppip set
    file_name2='eser-1µl1µA030_ACH.wav' #eser set
    _, regions1, _=AD.spect_loop(file_name1)
    _, regions2, _=AD.spect_loop(file_name2)
    #File 1
    img1=regions1[0][0]
    img2=regions1[1][0]
    img3=regions1[2][0]
    img4=regions1[3][0]
    img5=regions1[4][0]
    img6=regions1[5][0]
    img7=regions1[6][0]
    img8=regions1[7][0]
    img9=regions1[8][0]
    img10=regions1[9][0]
    img11=regions1[11][0]
    img12=regions1[12][0]
    img13=regions1[13][0]
    img14=regions1[14][0]
    img15=regions1[15][0]
    img16=regions1[16][0]
    img17=regions1[17][0]
    img18=regions1[18][0]
    img19=regions1[20][0]
    img20=regions1[21][0]
    img21=regions1[22][0]
    img22=regions1[23][0]
    img23=regions1[24][0]
    img24=regions1[25][0]
    img25=regions1[26][0]
    img26=regions1[27][0]
    img27=regions1[28][0]
    img28=regions1[29][0]
    img29=regions1[30][0]
    img30=regions1[31][0]
    img31=regions1[32][0]   
    img32=regions1[33][1]
    img33=regions1[34][0]
    img34=regions1[35][0]
    img35=regions1[36][0]
    img36=regions1[37][0]
    img37=regions1[38][0]
    img38=regions1[40][0]
    img39=regions1[41][0]   
    img40=regions1[42][0]
    img41=regions1[43][0]
    img42=regions1[44][0]
    img43=regions1[45][0]
    img44=regions1[49][0]
    img45=regions1[52][0]
    #File 2
    img46=regions2[1][0]
    img47=regions2[3][0]
    img48=regions2[4][0]
    img49=regions2[6][0]
    img50=regions2[7][0]
    img51=regions2[9][0]
    img52=regions2[11][0]
    img53=regions2[12][0]
    img54=regions2[14][0]
    img55=regions2[15][0]
    img56=regions2[17][0]
    img57=regions2[18][0]
    img58=regions2[19][0]
    img59=regions2[20][0]
    img60=regions2[22][0]
    img61=regions2[23][0]
    img62=regions2[24][0]
    img63=regions2[25][0]
    img63=regions2[28][0]
    
    templates_0={0: img1, 1: img2, 2: img3, 3: img4,
             4: img5, 5: img6, 6: img7, 7: img8,
             8: img9, 9: img10, 10: img11, 11: img12,
             12: img13, 13: img14, 14: img15, 15: img16,
             16: img17, 17: img18, 18: img19, 19: img20,
             20: img21, 21: img22, 22: img23, 23: img24,
             24: img25, 25: img26, 26: img27, 27: img28,
             28: img29, 29: img30, 30: img31, 31: img32,
             32: img33, 33: img34, 34: img35, 35: img36,
             36: img37, 37: img38, 38: img39, 39: img40,
             40: img41, 41: img42, 42: img43, 43: img44,
             44: img45}
    templates_1={0: img46, 1: img47, 2: img48, 3: img49,
             4: img50, 5: img51, 6: img52, 7: img53,
             8: img54, 9: img55, 10: img56, 11: img57,
             12: img58, 13: img59, 14: img60, 15: img61,
             16: img62, 17: img63}
    templates={0: templates_0, 1: templates_1}
    return(templates)


