def ModelBaseline_MultiClass(nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model






def ModelBaseline_OneVsRest(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))

    x_coh=Dense(nb_n1, activation='sigmoid')(inputs)
    x_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_mero=Dense(nb_n1, activation='sigmoid')(inputs)

    
    out_pre_coh=Dense(nb_n2, activation='sigmoid')(x_coh)
    out_pre_hyper=Dense(nb_n2, activation='sigmoid')(x_hyper)
    out_pre_mero=Dense(nb_n2, activation='sigmoid')(x_mero)

    
    out_coh=Dense(1, activation='sigmoid')(out_pre_coh)
    out_hyper=Dense(1, activation='sigmoid')(out_pre_hyper)
    out_mero=Dense(1, activation='sigmoid')(out_pre_mero)

    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared2=Dense(nb_n1, activation='sigmoid')(inputs)
    

    
    ind_coh=concatenate([inputs,x_shared1])
    ind_hyper=concatenate([inputs,x_shared1])
    ind_mero=concatenate([inputs,x_shared1])
  
    
    x_coh=Dense(nb_n2, activation='sigmoid')(ind_coh)
    x_hyper=Dense(nb_n2, activation='sigmoid')(ind_hyper)
    x_mero=Dense(nb_n2, activation='sigmoid')(ind_mero)

    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)

    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f])
    return model






def ModelOneVsRest_AllShared(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
   
    x_coh=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_hyper=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_mero=Dense(nb_n2, activation='sigmoid')(x_shared1)

    
    out_coh=Dense(1, activation='sigmoid')(x_coh)
    out_hyper=Dense(1, activation='sigmoid')(x_hyper)
    out_mero=Dense(1, activation='sigmoid')(x_mero)

    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f])
    return model







def ModelOneVsRest_SharedPrivate_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
 

    
    
    ind_coh1=concatenate([inputs,x_shared_coh_hyper])
    ind_coh2=concatenate([inputs,x_shared_coh_mero])
    
    
    ind_hyper1=concatenate([inputs,x_shared_hyper_mero])
    ind_hyper2=concatenate([inputs,x_shared_coh_hyper])
    
    
    ind_mero1=concatenate([inputs,x_shared_hyper_mero])
    ind_mero2=concatenate([inputs,x_shared_coh_mero])
    
   
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)

    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)
 
    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)

    
    

    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)

    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)

    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)

    

    
    
    out_coh_pre=concatenate([out_coh1,out_coh2])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2])
    
    out_mero_pre=concatenate([out_mero1,out_mero2])
    
 
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)

    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f])
    return model




def ModelOneVsRest_All_shared_2Per2(nb_n1=100,nb_n2=50):
    inputs = Input(shape=(input_dime,))
    
    x_shared_coh_hyper=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_coh_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    x_shared_hyper_mero=Dense(nb_n1, activation='sigmoid')(inputs)
    
    
    ind_coh1=x_shared_coh_hyper
    ind_coh2=x_shared_coh_mero

    
    ind_hyper1=x_shared_hyper_mero
    ind_hyper2=x_shared_coh_hyper
    
    ind_mero1=x_shared_hyper_mero
    ind_mero2=x_shared_coh_mero
  
    
    x_coh1=Dense(nb_n2, activation='sigmoid')(ind_coh1)
    x_coh2=Dense(nb_n2, activation='sigmoid')(ind_coh2)

    
    
    
    
    x_hyper1=Dense(nb_n2, activation='sigmoid')(ind_hyper1)
    x_hyper2=Dense(nb_n2, activation='sigmoid')(ind_hyper2)

    
    
    x_mero1=Dense(nb_n2, activation='sigmoid')(ind_mero1)
    x_mero2=Dense(nb_n2, activation='sigmoid')(ind_mero2)

    
   
    
    out_coh1=Dense(1, activation='sigmoid')(x_coh1)
    out_coh2=Dense(1, activation='sigmoid')(x_coh2)

    
    
    out_hyper1=Dense(1, activation='sigmoid')(x_hyper1)
    out_hyper2=Dense(1, activation='sigmoid')(x_hyper2)

    
    
    out_mero1=Dense(1, activation='sigmoid')(x_mero1)
    out_mero2=Dense(1, activation='sigmoid')(x_mero2)

    
    
   
    
    
    out_coh_pre=concatenate([out_coh1,out_coh2])
    
    out_hyper_pre=concatenate([out_hyper1,out_hyper2])
    
    out_mero_pre=concatenate([out_mero1,out_mero2])

    
    
    out_coh=Dense(1, activation='sigmoid')(out_coh_pre)
    out_hyper=Dense(1, activation='sigmoid')(out_hyper_pre)
    out_mero=Dense(1, activation='sigmoid')(out_mero_pre)

    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_coh,out_hyper,out_mero])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f])
    return model

