def MultiClass(nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model






def ModelBaseline_OneVsRest(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))

    x_coh=Dense(nb_n1, activation='sigmoid')(inputs)
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)
    out_pre_random=Dense(nb_n2, activation='sigmoid')(x_random)
    out_pre_syn=Dense(nb_n2, activation='sigmoid')(x_syn)
    
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)
    out_random=Dense(1, activation='sigmoid')(out_pre_random)
    out_syn=Dense(1, activation='sigmoid')(out_pre_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared2=Dense(nb_n1, activation='sigmoid')(inputs)
    

    
    ind_coh=concatenate([inputs,x_shared1])
    ind_hyper=concatenate([inputs,x_shared1])
    ind_mero=concatenate([inputs,x_shared1])
    ind_random=concatenate([inputs,x_shared1])
    ind_syn=concatenate([inputs,x_shared1])
    
    x_coh=Dense(nb_n2, activation='sigmoid')(ind_coh)
    x_hyper=Dense(nb_n2, activation='sigmoid')(ind_hyper)
    x_mero=Dense(nb_n2, activation='sigmoid')(ind_mero)
    x_random=Dense(nb_n2, activation='sigmoid')(ind_random)
    x_syn=Dense(nb_n2, activation='sigmoid')(ind_syn)
    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)
    out_random=Dense(1, activation='sigmoid')(x_random)
    out_syn=Dense(1, activation='sigmoid')(x_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model






def ModelOneVsRest_AllShared(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
   
    x_coh=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_hyper=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_mero=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_random=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_syn=Dense(nb_n2, activation='sigmoid')(x_shared1)
    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)
    out_random=Dense(1, activation='sigmoid')(x_random)
    out_syn=Dense(1, activation='sigmoid')(x_syn)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model







def ModelOneVsRest_SharedPrivate_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper])
    ind_coh2=concatenate([inputs,x_shared_coh_mero])
    ind_coh3=concatenate([inputs,x_shared_coh_random])
    ind_coh4=concatenate([inputs,x_shared_coh_syn])
    
    
    ind_hyper1=concatenate([inputs,x_shared_hyper_mero])
    ind_hyper2=concatenate([inputs,x_shared_hyper_random])
    ind_hyper3=concatenate([inputs,x_shared_hyper_syn])
    ind_hyper4=concatenate([inputs,x_shared_coh_hyper])
    
    
    ind_mero1=concatenate([inputs,x_shared_mero_random])
    ind_mero2=concatenate([inputs,x_shared_mero_syn])
    ind_mero3=concatenate([inputs,x_shared_hyper_mero])
    ind_mero4=concatenate([inputs,x_shared_coh_mero])
    
    
    ind_random1=concatenate([inputs,x_shared_random_syn])
    ind_random2=concatenate([inputs,x_shared_coh_random])
    ind_random3=concatenate([inputs,x_shared_hyper_random])
    ind_random4=concatenate([inputs,x_shared_mero_random])
    
    ind_syn1=concatenate([inputs,x_shared_coh_syn])
    ind_syn2=concatenate([inputs,x_shared_hyper_syn])
    ind_syn3=concatenate([inputs,x_shared_mero_syn])
    ind_syn4=concatenate([inputs,x_shared_random_syn])
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper4)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate_2Per2wORandom(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)

    x_random=Dense(nb_n1,activation='sigmoid')(inputs)
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper])
    ind_coh2=concatenate([inputs,x_shared_coh_mero])
    ind_coh3=concatenate([inputs,x_shared_coh_syn])
    
    
    ind_hyper1=concatenate([inputs,x_shared_hyper_mero])
    ind_hyper2=concatenate([inputs,x_shared_hyper_syn])
    ind_hyper3=concatenate([inputs,x_shared_coh_hyper])
    
    
    ind_mero1=concatenate([inputs,x_shared_mero_syn])
    ind_mero2=concatenate([inputs,x_shared_hyper_mero])
    ind_mero3=concatenate([inputs,x_shared_coh_mero])
    
    
    ind_syn1=concatenate([inputs,x_shared_coh_syn])
    ind_syn2=concatenate([inputs,x_shared_hyper_syn])
    ind_syn3=concatenate([inputs,x_shared_mero_syn])

    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)

    
    x_random1=Dense(nb_n2, activation='sigmoid')(x_random)

        
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)

    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)

    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
  
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3])
    
    out_random_pre=out_random1
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate_3Per3(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
   
    
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper_mero])
    ind_coh2=concatenate([inputs,x_shared_coh_hyper_random])
    ind_coh3=concatenate([inputs,x_shared_coh_hyper_syn])
    ind_coh4=concatenate([inputs,x_shared_coh_mero_random])
    ind_coh5=concatenate([inputs,x_shared_coh_mero_syn])
    ind_coh6=concatenate([inputs,x_shared_coh_random_syn])
    
    
    ind_hyper1=concatenate([inputs,x_shared_coh_hyper_mero])
    ind_hyper2=concatenate([inputs,x_shared_coh_hyper_random])
    ind_hyper3=concatenate([inputs,x_shared_coh_hyper_syn])
    ind_hyper4=concatenate([inputs,x_shared_hyper_mero_random])
    ind_hyper5=concatenate([inputs,x_shared_hyper_mero_syn])
    ind_hyper6=concatenate([inputs,x_shared_hyper_random_syn])
    
    
    ind_mero1=concatenate([inputs,x_shared_coh_hyper_mero])
    ind_mero2=concatenate([inputs,x_shared_coh_mero_random])
    ind_mero3=concatenate([inputs,x_shared_coh_mero_syn])
    ind_mero4=concatenate([inputs,x_shared_hyper_mero_random])
    ind_mero5=concatenate([inputs,x_shared_hyper_mero_syn])
    ind_mero6=concatenate([inputs,x_shared_mero_random_syn])
    
    
    ind_random1=concatenate([inputs,x_shared_mero_random_syn])
    ind_random2=concatenate([inputs,x_shared_coh_hyper_random])
    ind_random3=concatenate([inputs,x_shared_coh_mero_random])
    ind_random4=concatenate([inputs,x_shared_hyper_mero_random])
    ind_random5=concatenate([inputs,x_shared_coh_random_syn])
    ind_random6=concatenate([inputs,x_shared_hyper_random_syn])
    
    ind_syn1=concatenate([inputs,x_shared_hyper_mero_syn])
    ind_syn2=concatenate([inputs,x_shared_hyper_random_syn])
    ind_syn3=concatenate([inputs,x_shared_coh_hyper_syn])
    ind_syn4=concatenate([inputs,x_shared_mero_random_syn])
    ind_syn5=concatenate([inputs,x_shared_coh_mero_syn])
    ind_syn6=concatenate([inputs,x_shared_coh_random_syn])

    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    x_coh5=Dense(nb_n2, activation='sigmoid')(ind_coh5)
    x_coh6=Dense(nb_n2, activation='sigmoid')(ind_coh6)
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    x_hyper5=Dense(nb_n2, activation='sigmoid')(ind_hyper5)
    x_hyper6=Dense(nb_n2, activation='sigmoid')(ind_hyper6)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    x_mero5=Dense(nb_n2, activation='sigmoid')(ind_mero5)
    x_mero6=Dense(nb_n2, activation='sigmoid')(ind_mero6)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    x_random5=Dense(nb_n2, activation='sigmoid')(ind_random5)
    x_random6=Dense(nb_n2, activation='sigmoid')(ind_random6)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    x_syn5=Dense(nb_n2, activation='sigmoid')(ind_syn5)
    x_syn6=Dense(nb_n2, activation='sigmoid')(ind_syn6)

    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    out_coh5=Dense(1, activation='sigmoid')(x_coh5)
    out_coh6=Dense(1, activation='sigmoid')(x_coh6)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper4)
    out_hyper5=Dense(1, activation='sigmoid')(x_hyper5)
    out_hyper6=Dense(1, activation='sigmoid')(x_hyper6)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    out_mero5=Dense(1, activation='sigmoid')(x_mero5)
    out_mero6=Dense(1, activation='sigmoid')(x_mero6)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    out_random5=Dense(1, activation='sigmoid')(x_random5)
    out_random6=Dense(1, activation='sigmoid')(x_random6)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    out_syn5=Dense(1, activation='sigmoid')(x_syn5)
    out_syn6=Dense(1, activation='sigmoid')(x_syn6)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4,out_coh5,out_coh6])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4,out_hyper5,out_hyper6])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4,out_mero5,out_mero6])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4,out_random5,out_random6])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4,out_syn5,out_syn6])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate_4Per4(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
   
   
    
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper_mero_random])
    ind_coh2=concatenate([inputs,x_shared_coh_hyper_mero_syn])
    ind_coh3=concatenate([inputs,x_shared_coh_hyper_random_syn])
    ind_coh4=concatenate([inputs,x_shared_coh_mero_random_syn])
   
    
    
    ind_hyper1=concatenate([inputs,x_shared_coh_hyper_mero_random])
    ind_hyper2=concatenate([inputs,x_shared_coh_hyper_mero_syn])
    ind_hyper3=concatenate([inputs,x_shared_coh_hyper_random_syn])
    ind_hyper4=concatenate([inputs,x_shared_hyper_mero_random_syn])
    
    ind_mero1=concatenate([inputs,x_shared_coh_hyper_mero_random])
    ind_mero2=concatenate([inputs,x_shared_coh_hyper_mero_syn])
    ind_mero3=concatenate([inputs,x_shared_coh_mero_random_syn])
    ind_mero4=concatenate([inputs,x_shared_hyper_mero_random_syn])
    
    ind_random1=concatenate([inputs,x_shared_coh_hyper_mero_random])
    ind_random2=concatenate([inputs,x_shared_coh_hyper_random_syn])
    ind_random3=concatenate([inputs,x_shared_coh_mero_random_syn])
    ind_random4=concatenate([inputs,x_shared_hyper_mero_random_syn])

    ind_syn1=concatenate([inputs,x_shared_coh_hyper_mero_syn])
    ind_syn2=concatenate([inputs,x_shared_coh_hyper_random_syn])
    ind_syn3=concatenate([inputs,x_shared_coh_mero_random_syn])
    ind_syn4=concatenate([inputs,x_shared_hyper_mero_random_syn])
    

    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)

    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)

    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)

    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)

    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)


    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)

    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper4)

    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)

    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)

    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)

    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model




def ModelOneVsRest_All_shared_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    
    
    ind_coh1=x_shared_coh_hyper
    ind_coh2=x_shared_coh_mero
    ind_coh3=x_shared_coh_random
    ind_coh4=x_shared_coh_syn
    
    
    ind_hyper1=x_shared_hyper_mero
    ind_hyper2=x_shared_hyper_random
    ind_hyper3=x_shared_hyper_syn
    ind_hyper4=x_shared_coh_hyper
    
    
    ind_mero1=x_shared_mero_random
    ind_mero2=x_shared_mero_syn
    ind_mero3=x_shared_hyper_mero
    ind_mero4=x_shared_coh_mero
    
    
    ind_random1=x_shared_random_syn
    ind_random2=x_shared_coh_random
    ind_random3=x_shared_hyper_random
    ind_random4=x_shared_mero_random
    
    ind_syn1=x_shared_coh_syn
    ind_syn2=x_shared_hyper_syn
    ind_syn3=x_shared_mero_syn
    ind_syn4=x_shared_random_syn
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper3)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def ModelOneVsRest_AllShared_3Per3(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
   
    
    
    ind_coh1=x_shared_coh_hyper_mero
    ind_coh2=x_shared_coh_hyper_random
    ind_coh3=x_shared_coh_hyper_syn
    ind_coh4=x_shared_coh_mero_random
    ind_coh5=x_shared_coh_mero_syn
    ind_coh6=x_shared_coh_random_syn
    
    
    ind_hyper1=x_shared_coh_hyper_mero
    ind_hyper2=x_shared_coh_hyper_random
    ind_hyper3=x_shared_coh_hyper_syn
    ind_hyper4=x_shared_hyper_mero_random
    ind_hyper5=x_shared_hyper_mero_syn
    ind_hyper6=x_shared_hyper_random_syn
    
    
    ind_mero1=x_shared_coh_hyper_mero
    ind_mero2=x_shared_coh_mero_random
    ind_mero3=x_shared_coh_mero_syn
    ind_mero4=x_shared_hyper_mero_random
    ind_mero5=x_shared_hyper_mero_syn
    ind_mero6=x_shared_mero_random_syn
    
    
    ind_random1=x_shared_mero_random_syn
    ind_random2=x_shared_coh_hyper_random
    ind_random3=x_shared_coh_mero_random
    ind_random4=x_shared_hyper_mero_random
    ind_random5=x_shared_coh_random_syn
    ind_random6=x_shared_hyper_random_syn
    
    ind_syn1=x_shared_hyper_mero_syn
    ind_syn2=x_shared_hyper_random_syn
    ind_syn3=x_shared_coh_hyper_syn
    ind_syn4=x_shared_mero_random_syn
    ind_syn5=x_shared_coh_mero_syn
    ind_syn6=x_shared_coh_random_syn

    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)
    x_coh5=Dense(nb_n2, activation='sigmoid')(ind_coh5)
    x_coh6=Dense(nb_n2, activation='sigmoid')(ind_coh6)
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)
    x_hyper5=Dense(nb_n2, activation='sigmoid')(ind_hyper5)
    x_hyper6=Dense(nb_n2, activation='sigmoid')(ind_hyper6)
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)
    x_mero5=Dense(nb_n2, activation='sigmoid')(ind_mero5)
    x_mero6=Dense(nb_n2, activation='sigmoid')(ind_mero6)
    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)
    x_random5=Dense(nb_n2, activation='sigmoid')(ind_random5)
    x_random6=Dense(nb_n2, activation='sigmoid')(ind_random6)
    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)
    x_syn5=Dense(nb_n2, activation='sigmoid')(ind_syn5)
    x_syn6=Dense(nb_n2, activation='sigmoid')(ind_syn6)

    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)
    out_coh5=Dense(1, activation='sigmoid')(x_coh5)
    out_coh6=Dense(1, activation='sigmoid')(x_coh6)
    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper4)
    out_hyper5=Dense(1, activation='sigmoid')(x_hyper5)
    out_hyper6=Dense(1, activation='sigmoid')(x_hyper6)
    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)
    out_mero5=Dense(1, activation='sigmoid')(x_mero5)
    out_mero6=Dense(1, activation='sigmoid')(x_mero6)
    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)
    out_random5=Dense(1, activation='sigmoid')(x_random5)
    out_random6=Dense(1, activation='sigmoid')(x_random6)
    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)
    out_syn5=Dense(1, activation='sigmoid')(x_syn5)
    out_syn6=Dense(1, activation='sigmoid')(x_syn6)
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4,out_coh5,out_coh6])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4,out_hyper5,out_hyper6])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4,out_mero5,out_mero6])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4,out_random5,out_random6])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4,out_syn5,out_syn6])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def ModelOneVsRest_AllShared_4Per4(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper_mero_random=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_mero_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_hyper_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero_random_syn=Dense(nb_n1, activation='sigmoid')(inputs)
   
   
    
    
    ind_coh1=x_shared_coh_hyper_mero_random
    ind_coh2=x_shared_coh_hyper_mero_syn
    ind_coh3=x_shared_coh_hyper_random_syn
    ind_coh4=x_shared_coh_mero_random_syn
   
    
    
    ind_hyper1=x_shared_coh_hyper_mero_random
    ind_hyper2=x_shared_coh_hyper_mero_syn
    ind_hyper3=x_shared_coh_hyper_random_syn
    ind_hyper4=x_shared_hyper_mero_random_syn
    
    ind_mero1=x_shared_coh_hyper_mero_random
    ind_mero2=x_shared_coh_hyper_mero_syn
    ind_mero3=x_shared_coh_mero_random_syn
    ind_mero4=x_shared_hyper_mero_random_syn
    
    ind_random1=x_shared_coh_hyper_mero_random
    ind_random2=x_shared_coh_hyper_random_syn
    ind_random3=x_shared_coh_mero_random_syn
    ind_random4=x_shared_hyper_mero_random_syn

    ind_syn1=x_shared_coh_hyper_mero_syn
    ind_syn2=x_shared_coh_hyper_random_syn
    ind_syn3=x_shared_coh_mero_random_syn
    ind_syn4=x_shared_hyper_mero_random_syn
    

    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)
    x_coh3=Dense(nb_n2, activation='sigmoid')(ind_coh3)
    x_coh4=Dense(nb_n2, activation='sigmoid')(ind_coh4)

    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
    x_hyper3=Dense(nb_n2, activation='sigmoid')(ind_hyper3)
    x_hyper4=Dense(nb_n2, activation='sigmoid')(ind_hyper4)

    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)
    x_mero3=Dense(nb_n2, activation='sigmoid')(ind_mero3)
    x_mero4=Dense(nb_n2, activation='sigmoid')(ind_mero4)

    
    
    x_random1=Dense(nb_n2, activation='sigmoid')(ind_random1)
    x_random2=Dense(nb_n2, activation='sigmoid')(ind_random2)
    x_random3=Dense(nb_n2, activation='sigmoid')(ind_random3)
    x_random4=Dense(nb_n2, activation='sigmoid')(ind_random4)

    
    
    x_syn1=Dense(nb_n2, activation='sigmoid')(ind_syn1)
    x_syn2=Dense(nb_n2, activation='sigmoid')(ind_syn2)
    x_syn3=Dense(nb_n2, activation='sigmoid')(ind_syn3)
    x_syn4=Dense(nb_n2, activation='sigmoid')(ind_syn4)


    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)
    out_coh3=Dense(1, activation='sigmoid')(x_coh3)
    out_coh4=Dense(1, activation='sigmoid')(x_coh4)

    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)
    out_hyper3=Dense(1, activation='sigmoid')(x_hyper3)
    out_hyper4=Dense(1, activation='sigmoid')(x_hyper4)

    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)
    out_mero3=Dense(1, activation='sigmoid')(x_mero3)
    out_mero4=Dense(1, activation='sigmoid')(x_mero4)

    
    
    out_random1=Dense(1, activation='sigmoid')(x_random1)
    out_random2=Dense(1, activation='sigmoid')(x_random2)
    out_random3=Dense(1, activation='sigmoid')(x_random3)
    out_random4=Dense(1, activation='sigmoid')(x_random4)

    
    out_syn1=Dense(1, activation='sigmoid')(x_syn1)
    out_syn2=Dense(1, activation='sigmoid')(x_syn2)
    out_syn3=Dense(1, activation='sigmoid')(x_syn3)
    out_syn4=Dense(1, activation='sigmoid')(x_syn4)

    
    
    out_coh_pre=concatenate([out_coh1,out_coh2,out_coh3,out_coh4])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2,out_hyper3,out_hyper4])
    
    out_mero_pre=concatenate([out_mero1,out_mero2,out_mero3,out_mero4])
    
    out_random_pre=concatenate([out_random1,out_random2,out_random3,out_random4])
    
    out_syn_pre=concatenate([out_syn1,out_syn2,out_syn3,out_syn4])
    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)
    out_random=Dense(1, activation='sigmoid')(out_random_pre)
    out_syn=Dense(1, activation='sigmoid')(out_syn_pre)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero,out_random,out_syn])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model