def Model_OneVsRest_Hier1_0(nb_n1=100,nb_n2=50):
    #A chaque fois on fait passer la couche cachée de la couche en question + les outputs successifs à la couche suivante

    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_mero=concatenate([inputs,x_random,out_random])
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs_mero)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    
    inputs_hyper=concatenate([inputs,x_mero,out_random,out_mero])
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs_hyper)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    
    inputs_coh=concatenate([inputs,x_hyper,out_random,out_mero,out_hyper])
    x_coh=Dense(nb_n1, activation='sigmoid')(inputs_coh)
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
    inputs_syn=concatenate([inputs,x_coh,out_random,out_mero,out_hyper,out_coh])
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs_syn)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    


    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model




def Model_OneVsRest_Hier1_1(nb_n1=100,nb_n2=50):
    #Exactement comme le 1_0 juste quen ici on fait passer que la couche cachée sans l'output.
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_mero=concatenate([inputs,x_random])
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs_mero)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    
    inputs_hyper=concatenate([inputs,x_mero])
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs_hyper)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    
    inputs_coh=concatenate([inputs,x_hyper])
    x_coh=Dense(nb_n1, activation='sigmoid')(inputs_coh)
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
    inputs_syn=concatenate([inputs,x_coh])
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs_syn)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
    
    
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def Model_OneVsRest_Hier1_1RMSCH(nb_n1=100,nb_n2=50):
    #Exactement comme le 1_0 juste quen ici on fait passer que la couche cachée sans l'output.
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_mero=concatenate([inputs,x_random])
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs_mero)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)

    inputs_syn=concatenate([inputs,x_mero])
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs_syn)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)

    inputs_coh=concatenate([inputs,x_syn])
    x_coh=Dense(nb_n1, activation='sigmoid')(inputs_coh)
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)

    
    
    inputs_hyper=concatenate([inputs,x_coh])
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs_hyper)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def Model_OneVsRest_Hier1_1RMCSH(nb_n1=100,nb_n2=50):
    #Exactement comme le 1_0 juste quen ici on fait passer que la couche cachée sans l'output.
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_mero=concatenate([inputs,x_random])
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs_mero)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)

    inputs_coh=concatenate([inputs,x_mero])
    x_coh=Dense(nb_n1, activation='sigmoid')(inputs_coh)
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)

    inputs_syn=concatenate([inputs,x_coh])
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs_syn)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
    inputs_hyper=concatenate([inputs,x_syn])
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs_hyper)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)

    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def Model_OneVsRest_Hier1_2(nb_n1=100,nb_n2=50):
    #Exactement comme le 1_0 juste quen ici on fait passer que la couche cachée avec l'output du random
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    
    
    inputs_mero=concatenate([inputs,out_random])
    
    
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs_mero)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    
    inputs_hyper=concatenate([inputs,x_mero,out_random])
    
    
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs_hyper)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    
    inputs_coh=concatenate([inputs,x_hyper,out_random])
    
    
    x_coh=Dense(nb_n1, activation='sigmoid')(inputs_coh)
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
    inputs_syn=concatenate([inputs,x_coh,out_random])
    
    
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs_syn)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
   

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def Model_OneVsRest_Hier2_0(nb_n1=100,nb_n2=50):
    #lhierarchie se fait juste entre random puis le reste. on résout d'abord le random puis le restec'est du shared private oneVsRest
    inputs = Input(shape=(600,))
    #Partie Random
    x_shared=Dense(nb_n1, activation='sigmoid')
    x=x_shared(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_All=concatenate([inputs,x,out_random])
    
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(inputs_All)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    

    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(inputs_All)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
 

    out_pre_coh=Dense(nb_n2, activation='sigmoid')(inputs_All)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    

    out_pre_syn=Dense(nb_n2, activation='sigmoid')(inputs_All)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
   

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model

def Model_OneVsRest_Hier2_1(nb_n1=100,nb_n2=50):
    #comme celui d'avant mais le shared se fait sans le random. il se fait apres resolution du random. puis on rajoute out_random to all
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_All=concatenate([inputs,out_random])
    x_shared=Dense(nb_n1, activation='sigmoid')(inputs_All)
    
    
    inputs_All2=concatenate([inputs,x_shared])
    
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    

    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    

    out_pre_coh=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
 
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
    out_final_random=concatenate([out_coh,out_hyper,out_mero,out_random,out_syn])
    out_f_random=Dense(1, activation='sigmoid')(out_final_random)
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_f_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model

def Model_OneVsRest_Hier2_2(nb_n1=100,nb_n2=50):
    ##comme celui d'avant mais à chaque fois on rajoute l'ioutput précédent.
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_All=concatenate([inputs,out_random])
    x_shared=Dense(nb_n1, activation='sigmoid')(inputs_All)
    
    
    inputs_All2=concatenate([inputs,x_shared])
    
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    

    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero]))
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    

    out_pre_coh=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero,out_hyper]))
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
 
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero,out_hyper,out_coh]))
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
   

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def Model_OneVsRest_Hier2_3(nb_n1=100,nb_n2=50):
    #Exactement comme celui d'avant mais l'out du random se fait aussi grace aux autres outputs
    inputs = Input(shape=(600,))
    #Partie Random
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    
    inputs_All=concatenate([inputs,out_random])
    x_shared=Dense(nb_n1, activation='sigmoid')(inputs_All)
    
    
    inputs_All2=concatenate([inputs,x_shared])
    
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(inputs_All2)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    

    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero]))
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    

    out_pre_coh=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero,out_hyper]))
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    
 
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(concatenate([inputs_All2,out_mero,out_hyper,out_coh]))
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    
   
    out_final_random=concatenate([out_coh,out_hyper,out_mero,out_random,out_syn])
    out_f_random=Dense(1, activation='sigmoid')(out_final_random)
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_f_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model