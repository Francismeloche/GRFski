# -*- coding: utf-8 -*-

'''
Created June 10th 2022
Modified December 18th 2024

@author : Francis Meloche (UQAR)
@author :  Louis Guillet INRIA

set of functions to create virtual metastable ski slope to assess the skier triggering probability
The virtual ski slope is set to 35 degree
The skier loading (R) is set to 780 N from Gaume and Reuter 2017

The slab depth is spatially variable using a Gaussian Random Field GRF using 3 parameters:
    1) Mean slab depth
    2) Variance slab depth
    3) Correlation length slab depth

The slab density and elastic modulus are set to the mean slab depth vi empirical relationship
    1) Slab density McClung 2009
    2) Elastic modulus Sigrist 2006
    
The weak layer shear strength is set to the local slab depth using an derived empirical/Mohr-Colomb relation
    1) Weak layer shear strength Gaume and Reuter 2017

'''
import numpy as np
import gstools as gs

from numba import njit,prange,jit,config # important for parallelisation of the code


def Sk_index(Ac,lsk):
    if lsk == 0:
        Sp = 0
    else:
        Sp = Ac/lsk
    return Sp


@njit(nopython = True)
def secant_zero(f,tp,t,phi,D,R, p0, p1, max_iterations=1500, tolerence=1e-10):

    q0 = f(p0,tp,t,phi,D,R)
    q1 = f(p1,tp,t,phi,D,R)

    for _ in prange(max_iterations):
        p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if np.abs(p - p1) < tolerence:
            return p
        p0 = p1
        q0 = q1
        p1 = p
        q1 = f(p,tp,t,phi,D,R)

    return p1

@njit(nopython=True)
def sk38_f(Tp,thau,H_trans):
    R = 780
    a = np.radians(54)
    delta_thau = (2*R)/(np.pi*H_trans)*(np.cos(a)*(np.sin(a)**2)*np.sin(a+np.radians(38)))
    sk38 = Tp/(thau+delta_thau)
    return sk38
    
@njit(nopython=True)
def lsk(H,alpha):
    lsk = H*((1/np.tan(np.abs(np.radians(alpha[0])))) - (1/np.tan(np.abs(np.radians(alpha[1])))))
    return lsk

@njit(nopython=True)
def find_alpha(a,tp,t,phi,D,R):
    """
    Fonction pour trouver les deux angles alpha selon l`équation 2 de gaume et reuter 2017
    en trouvant les racines de l`équation 1
    """
    a = np.radians(a)
    delta_thau = (2*R*np.cos(a)*(np.sin(a)**2)*np.sin(a+phi))/(np.pi*D)
    return t + delta_thau - tp


config.THREADING_LAYER = 'forksafe'

@njit(parallel = True)
def lsk_ongrid2(H_trans,thau,Tp,Ac):
    lsk_grid = np.zeros(H_trans.shape)
    Sp_grid = np.zeros(H_trans.shape)
    
    R = 780
    g = 9.81
    phi = np.radians(35)

    for i in prange(H_trans.shape[0]):
        for j in prange(H_trans.shape[1]):
            sk38 = sk38_f(Tp[i,j],thau[i,j],H_trans[i,j])
            if thau[i,j] > Tp[i,j]:
                lsk1 = 0
            elif sk38 > 0.99:
                lsk1 = 0
            else:
                alpha1 = secant_zero(find_alpha,Tp[i,j],thau[i,j],phi,H_trans[i,j],R,0,54)
                alpha2 = secant_zero(find_alpha,Tp[i,j],thau[i,j],phi,H_trans[i,j],R,54,90)
                alpha = [alpha1,alpha2]
                if np.isnan(alpha[0]) == True:
                    lsk1 = 0
                elif alpha[0] < 1:
                    lsk1 = 0
                elif alpha[0] > 90:
                    lsk1 = 0
                elif np.isnan(alpha[1]) == True:
                    lsk1 = 0
                elif alpha[1] < 1:
                    lsk1 = 0
                elif alpha[1] > 90:
                    lsk1 = 0
                else:
                    lsk1 = lsk(H_trans[i,j],alpha)      
            if lsk1 < 0.01:
                lsk1 = 0
                lsk_grid[i,j] = lsk1
                Sp = 2
            elif lsk1 > 0.01:
                lsk_grid[i,j] = lsk1
                Sp = Ac[i,j]/lsk1
            else:
                Sp = 2
            if Sp > 2:
                Sp = 2
                Sp_grid[i,j] = Sp
            else:
                Sp_grid[i,j] = Sp
        
    return lsk_grid,Sp_grid

@njit(parallel=True)
def skieur_proba(Sp_grid,nb_skieur,space_skieur,rayon_ski):
    """
    space_skier est un paramètre à donner en m
    rayon de skieur également en m
    nb de trace de skieur en sinus
    """
    j_coord = np.linspace(0,1000,501)
    i_coord = np.linspace(0,2000,1001) # 100m en dm (1000) car résolution 0.1m
    i_arr = np.ones(Sp_grid.shape)*i_coord
    j_trans = j_coord.reshape(501,1)
    j_arr = np.ones(Sp_grid.shape)*j_trans
    
    skieur_grid = np.zeros(Sp_grid.shape)
    start = 0
    nb_hits = 0
    amplitude_skidm = rayon_ski*10*4 
    rayon_skidm = 50
        
    b = space_skieur*10

    for sk in range(0,nb_skieur,1):
        if sk == 0:
            b = 0
        elif sk == 1:
            b = space_skieur*10
        b = b + space_skieur*10
        i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (start + b)
        sksin_grid1 = np.where(i_arr>i1,1,0)

        i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (start + 10 + b)
        sksin_grid2 = np.where(i_arr>i2,0,1)

        skieur1 = sksin_grid1*sksin_grid2
        skieur_trigger = skieur1/Sp_grid
        sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
        sum_trigger = sum_trigger_arr.sum()

        if sum_trigger > 0:
            hit = 1
            nb_hits = nb_hits + hit 

        skieur_grid = skieur_grid + skieur1
    proba_trigger = nb_hits/nb_skieur
    #print("probabilité de trigger :",proba_trigger)

    trigger_grid = skieur_grid / Sp_grid
    return trigger_grid, proba_trigger

def gen_GRF2d_arr_pslab (seeds,mean,var,len_scale):
    x = np.linspace(0,200,1001) # 100 m de largeur au 10 cm
    y = np.linspace(0,100,501) # 50m de largeur au 10 cm
    
    model = gs.Gaussian(dim=2, var = var, len_scale = len_scale)
    srf = gs.field.SRF(model,mean = mean, seed=200519)

    field = srf.structured([x, y],seed = seeds, store=f"field{seeds}")
    #print("mean:", mean, " variance:", var, " correlation length:",len_scale)
    #srf.plot()
    
    H_trans = np.transpose(field)
    
    #array H_moyenne
    H_mean = np.ones(H_trans.shape)*mean
    
    P_slab = 100+135*H_mean**0.4  # McClung 2009
    #E_slab = ((5.07*10**9)*(P_slab/917)**5.13)/1e+6
    E_slab = ((9.68*10**8)*(P_slab/917)**2.94)/1e+6 # sigrist 2006
    
    g = 9.81
    phi = 35
    psi = 27
    c = 300
    
    thau = P_slab*g*H_trans*np.sin(np.radians(phi))
    sigma = P_slab*g*H_trans*np.cos(np.radians(phi))

    #Tp = c + A*g*np.cos(np.radians(phi))*np.tan(np.radians(psi))*H_trans**(1+B)
    Tp = c + 1370*H_trans**1.3
    #Tp = 1450
    #Tp = c + 100*g*np.cos(np.radians(phi))*np.tan(np.radians(psi))*H_trans+135*g*np.cos(np.radians(phi))*np.tan(np.radians(psi))*H_trans**(1+B)

    # Calcul de Ac
    v = 0.2 #ratio de poisson
    Gwl = 0.2 #Mpa
    Dwl = 0.04
    Eprime = E_slab/(1-v**2)
    cl_scale = np.sqrt(Eprime*H_trans*Dwl/Gwl)
    Ac = cl_scale*((-thau + np.sqrt(thau**2 + 2*sigma*(Tp-thau)))/sigma)
    
    
    return H_trans,P_slab,E_slab,thau,Tp,Ac

def gen_probaskieur_multireal(ens_setvar,nb_realisations):
    dist_probaGRF = {} #stock distribution probability dans un dictionnaire
    #H_GRF = {} #
    #skieurtrigger_GRF = {}
    m_std_probaGRF = {}
    for set_var in ens_setvar:
        probaskieur_list = []
        H_list = []
        skieur_list = []
        mean_slab = set_var[0]
        var_slab = set_var[1]
        lscale_slab = set_var[2]
        rayon_skieur = set_var[3]
        space_skieur = set_var[4]
        l_rayon_ratio = set_var[5]
        l_space_ratio = set_var[6]
        nb_skieur = int(200/space_skieur)
        
        ens_no = nb_realisations
        for seed in range(ens_no):
            #print(seed)
            H_trans,p_grid,E_slab,thau,Tp,Ac = gen_GRF2d_arr_pslab(seed,mean_slab,var_slab,lscale_slab)
            lsk_grid,Sp_grid = lsk_ongrid2(H_trans,thau,Tp,Ac)
            skieur_grid,proba_trigger = skieur_proba(Sp_grid,nb_skieur,space_skieur,rayon_skieur)
            probaskieur_list.append(proba_trigger)
            H_list.append(H_trans)
            skieur_list.append(skieur_grid)
        GRF_str = "m" + str(mean_slab) + "_var" + str(var_slab) + "_l" + str(lscale_slab) + "_R" + str(rayon_skieur) + "_S" + str(space_skieur)
        mean_proba = np.mean(probaskieur_list)
        std_proba = np.std(probaskieur_list)
        m_std_probaGRF.update({GRF_str:[mean_slab,var_slab,lscale_slab,rayon_skieur,space_skieur,nb_skieur,l_rayon_ratio,l_space_ratio,mean_proba,std_proba]})
        dist_GRF = {GRF_str:probaskieur_list}
        dist_probaGRF.update(dist_GRF)
        
        print("GRF parameters set:" + GRF_str)
        print("Probability of skier triggering, mean:" + str(mean_proba) + " std:" +  str(std_proba))
        #H_dict = {GRF_str:H_list}
        #H_GRF.update(H_dict)
        
        #skieur_GRF = {GRF_str:skieur_list}
        #skieurtrigger_GRF.update(skieur_GRF)
    return dist_probaGRF,m_std_probaGRF#, H_GRF, skieurtrigger_GRF,

@njit(parallel=True)
def skieur_addgroup_alea(Sp_grid, space_skieur,rayon_ski,random_position_sk1):
    """
    space_skier est un paramètre à donner en m
    rayon de skieur également en m
    nb de trace de skieur en sinus
    """
    j_coord = np.linspace(0,1000,1001)
    i_coord = np.linspace(0,2000,2001) # 100m en dm (1000) car résolution 0.1m
    i_arr = np.ones(Sp_grid.shape)*i_coord
    j_trans = j_coord.reshape(1001,1)
    j_arr = np.ones(Sp_grid.shape)*j_trans
    
    skieur_grid = np.zeros(Sp_grid.shape)
    #start = 0
    nb_hits = 0
    amplitude_skidm = rayon_ski*10*4 
    rayon_skidm = 50
        
    #random_position_sk1 = np.random.uniform(50,1900) # applique un tampon de 50 m à gauche et 100m à droite pour fitter 2 skieur
    
 
    #for sk in range(0,1,1): # seulement 2 skieur
    #    if sk == 0:
    #        b = 0
    #    elif sk == 1:
    b = 0
    b = b + space_skieur*10
    i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + b)
    sksin_grid1 = np.where(i_arr>i1,1,0)

    i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 + b)
    sksin_grid2 = np.where(i_arr>i2,0,1)

    skieur1 = sksin_grid1*sksin_grid2
    skieur_trigger = skieur1/Sp_grid
    sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
    sum_trigger = sum_trigger_arr.sum()

    if sum_trigger > 0:
        hit = 1
        nb_hits = nb_hits + hit 

    skieur_grid = skieur_grid + skieur1
    #print("nb_hits:", nb_hits)

    
    
    #Si les deux skieurs ont pas déclenchés on rajoute des skieurs aleatoirement sur la pente
    if nb_hits == 0:
        #print(" 2 skiers track with no hit")
        #print("Adding random skier")
        loop_count = 0
        nb_skier = 0
        while nb_hits == 0:
            if loop_count > 50:
                break
            else:
                random_position_sk = np.random.uniform(50,1950)

                i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk)
                sksin_grid1 = np.where(i_arr>i1,1,0)

                i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk + 10)
                sksin_grid2 = np.where(i_arr>i2,0,1)

                skieur1 = sksin_grid1*sksin_grid2
                skieur_trigger = skieur1/Sp_grid
                sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                sum_trigger = sum_trigger_arr.sum()

                if sum_trigger > 0:
                    hit = 1
                    nb_hits = nb_hits + hit
                skieur_grid = skieur_grid + skieur1
                nb_skier += 1 
                loop_count += 1
                
        #print("nb of aditionnal random skier", nb_skier)    
        trigger_grid_alea = skieur_grid / Sp_grid
    #else:
    #    print("fuck")
    return trigger_grid_alea, nb_skier

@njit(parallel=True)
def skieur_addgroup_struct(Sp_grid, space_skieur,rayon_ski,random_position_sk1):
    """
    space_skier est un paramètre à donner en m
    rayon de skieur également en m
    nb de trace de skieur en sinus
    """
    j_coord = np.linspace(0,1000,1001)
    i_coord = np.linspace(0,2000,2001) # 100m en dm (1000) car résolution 0.1m
    i_arr = np.ones(Sp_grid.shape)*i_coord
    j_trans = j_coord.reshape(1001,1)
    j_arr = np.ones(Sp_grid.shape)*j_trans
    
    skieur_grid = np.zeros(Sp_grid.shape)
    #start = 0
    nb_hits = 0
    amplitude_skidm = rayon_ski*10*4 
    rayon_skidm = 50
        
    #random_position_sk1 = np.random.randint(10,1990) # applique un tampon de 50 m à gauche et 100m à droite pour fitter 2 skieur
    #print(random_position_sk1)

        #random_position_sk1 = np.random.randint(50,1900) # applique un tampon de 50 m à gauche et 100m à droite pour fitter 2 skieur
    #print(random_position_sk1)

   # for sk in range(0,1,1): # seulement 2 skieur
   #     if sk == 0:
   #         b1 = 0
   #     elif sk == 1:
    b1 = 0
    i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + b1)
    sksin_grid1 = np.where(i_arr>i1,1,0)

    i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 + b1)
    sksin_grid2 = np.where(i_arr>i2,0,1)

    skieur = sksin_grid1*sksin_grid2
    skieur_trigger = skieur/Sp_grid
    sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
    sum_trigger = sum_trigger_arr.sum()

    if sum_trigger > 0:
        hit = 1
        nb_hits = nb_hits + hit 

    skieur_grid = skieur_grid + skieur
    #print("nb_hits:", nb_hits)
    
    if nb_hits == 0:
        loop_count = 0
        nb_skier = 0
        nb_hits2 = 0
        while nb_hits2 == 0:
            #print(loop_count,"loop_count")
            if loop_count > 50:
                break
            else:
                if loop_count == 0:
                    b = space_skieur*10
                #elif loop_count == 1:
                #    b = space_skieur*10
                if random_position_sk1 > 1800:
                    #print("recule")
                    #print(b, "b")
                    i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 - b)
                    sksin_grid1 = np.where(i_arr>i1,1,0)

                    i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 - b)
                    sksin_grid2 = np.where(i_arr>i2,0,1)
                    b = b + space_skieur*10
                    #print(b, "b")

                    skieur1 = sksin_grid1*sksin_grid2
                    skieur_trigger = skieur1/Sp_grid
                    sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                    sum_trigger = sum_trigger_arr.sum()
                    nb_skier += 1 
                elif random_position_sk1 < 200:
                    #print("avance")
                    #print(b, "b")
                    i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + b)
                    sksin_grid1 = np.where(i_arr>i1,1,0)

                    i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 + b)
                    sksin_grid2 = np.where(i_arr>i2,0,1)
                    b = b + space_skieur*10
                    #print(b, "b")

                    skieur1 = sksin_grid1*sksin_grid2
                    skieur_trigger = skieur1/Sp_grid
                    sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                    sum_trigger = sum_trigger_arr.sum()
                    nb_skier += 1 
                else:
                    if (loop_count % 2) == 0:
                        #print("alternance")
                        if (random_position_sk1 - b) > 50:
                            #print(b,"paire")
                            i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 - b)
                            sksin_grid1 = np.where(i_arr>i1,1,0)

                            i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 - b)
                            sksin_grid2 = np.where(i_arr>i2,0,1)
                            #b = b + space_skieur*10
                            #print(b,"paire")

                            skieur1 = sksin_grid1*sksin_grid2
                            skieur_trigger = skieur1/Sp_grid
                            sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                            sum_trigger = sum_trigger_arr.sum()
                            nb_skier += 1
                        else:
                            i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 - b)
                            sksin_grid1 = np.where(i_arr>i1,1,0)

                            i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 - b)
                            sksin_grid2 = np.where(i_arr>i2,0,1)
                            #b = b + space_skieur*10

                            skieur1 = sksin_grid1*sksin_grid2
                            skieur_trigger = skieur1/Sp_grid
                            sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                            sum_trigger = sum_trigger_arr.sum()

                    else :
                        if (random_position_sk1 + b) < 1950:
    
                            #print(b,"impaire")
                            i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + b)
                            sksin_grid1 = np.where(i_arr>i1,1,0)

                            i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 + b)
                            sksin_grid2 = np.where(i_arr>i2,0,1)
                            #b = b + space_skieur*10
                            #print(b,"impaire")

                            skieur1 = sksin_grid1*sksin_grid2
                            skieur_trigger = skieur1/Sp_grid
                            sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                            sum_trigger = sum_trigger_arr.sum()
                            nb_skier += 1
                        else:
                            i1=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 - b)
                            sksin_grid1 = np.where(i_arr>i1,1,0)

                            i2=np.sin(((2*np.pi)/amplitude_skidm)*j_arr)*rayon_skidm + (random_position_sk1 + 10 - b)
                            sksin_grid2 = np.where(i_arr>i2,0,1)
                            #b = b + space_skieur*10

                            skieur1 = sksin_grid1*sksin_grid2
                            skieur_trigger = skieur1/Sp_grid
                            sum_trigger_arr = np.where(skieur_trigger > 1,1,0)
                            sum_trigger = sum_trigger_arr.sum()

                            #print("no skier")
                            #print("skier is outside of the slope")
                        b = b + space_skieur *10
                    
                if sum_trigger > 0:
                    hit = 1
                    nb_hits2 = nb_hits2 + hit
                skieur_grid = skieur_grid + skieur1
                #nb_skier += 1 
                loop_count += 1
        if nb_hits2 < 1:
            nb_skier = 50
            #print("no trigger")
                
    #print("nb of aditionnal skier in the group", nb_skier)    
        trigger_grid_struct = skieur_grid / Sp_grid
    #else:
    #    print("fuck")
    return trigger_grid_struct, nb_skier
