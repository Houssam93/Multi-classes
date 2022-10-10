def ModelBaseline_MultiClass(nb_n1=100,nb_n2=50):
    """Defines the NN baseline.
    Two hidden layers, followed by the output layer. 
    """
    model = Sequential()
    model.add(Dense(nb_n1, activation='sigmoid', input_dim=input_dime))
    model.add(Dense(nb_n2, activation='sigmoid'))
    model.add(Dense(16, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy')
    return model






def ModelBaseline_OneVsRest(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))

    x_1=Dense(nb_n1, activation='sigmoid')(inputs)
    x_2=Dense(nb_n1, activation='sigmoid')(inputs)
    x_3=Dense(nb_n1, activation='sigmoid')(inputs)
    x_4=Dense(nb_n1, activation='sigmoid')(inputs)
    x_5=Dense(nb_n1, activation='sigmoid')(inputs)
    x_6=Dense(nb_n1, activation='sigmoid')(inputs)
    x_7=Dense(nb_n1, activation='sigmoid')(inputs)
    x_8=Dense(nb_n1, activation='sigmoid')(inputs)
    x_9=Dense(nb_n1, activation='sigmoid')(inputs)
    x_10=Dense(nb_n1, activation='sigmoid')(inputs)
    x_11=Dense(nb_n1, activation='sigmoid')(inputs)
    x_12=Dense(nb_n1, activation='sigmoid')(inputs)
    x_13=Dense(nb_n1, activation='sigmoid')(inputs)
    x_14=Dense(nb_n1, activation='sigmoid')(inputs)
    x_15=Dense(nb_n1, activation='sigmoid')(inputs)
    x_16=Dense(nb_n1, activation='sigmoid')(inputs)
    
    out_pre_1=Dense(nb_n2, activation='sigmoid')(x_1)
    out_pre_2=Dense(nb_n2, activation='sigmoid')(x_2)
    out_pre_3=Dense(nb_n2, activation='sigmoid')(x_3)
    out_pre_4=Dense(nb_n2, activation='sigmoid')(x_4)
    out_pre_5=Dense(nb_n2, activation='sigmoid')(x_5)
    out_pre_6=Dense(nb_n2, activation='sigmoid')(x_6)
    out_pre_7=Dense(nb_n2, activation='sigmoid')(x_7)
    out_pre_8=Dense(nb_n2, activation='sigmoid')(x_8)
    out_pre_9=Dense(nb_n2, activation='sigmoid')(x_9)
    out_pre_10=Dense(nb_n2, activation='sigmoid')(x_10)
    out_pre_11=Dense(nb_n2, activation='sigmoid')(x_11)
    out_pre_12=Dense(nb_n2, activation='sigmoid')(x_12)
    out_pre_13=Dense(nb_n2, activation='sigmoid')(x_13)
    out_pre_14=Dense(nb_n2, activation='sigmoid')(x_14)
    out_pre_15=Dense(nb_n2, activation='sigmoid')(x_15)
    out_pre_16=Dense(nb_n2, activation='sigmoid')(x_16)
    
    out_1=Dense(1, activation='sigmoid')(out_pre_1)
    out_2=Dense(1, activation='sigmoid')(out_pre_2)
    out_3=Dense(1, activation='sigmoid')(out_pre_3)
    out_4=Dense(1, activation='sigmoid')(out_pre_4)
    out_5=Dense(1, activation='sigmoid')(out_pre_5)
    out_6=Dense(1, activation='sigmoid')(out_pre_6)
    out_7=Dense(1, activation='sigmoid')(out_pre_7)
    out_8=Dense(1, activation='sigmoid')(out_pre_8)
    out_9=Dense(1, activation='sigmoid')(out_pre_9)
    out_10=Dense(1, activation='sigmoid')(out_pre_10)
    out_11=Dense(1, activation='sigmoid')(out_pre_11)
    out_12=Dense(1, activation='sigmoid')(out_pre_12)
    out_13=Dense(1, activation='sigmoid')(out_pre_13)
    out_14=Dense(1, activation='sigmoid')(out_pre_14)
    out_15=Dense(1, activation='sigmoid')(out_pre_15)
    out_16=Dense(1, activation='sigmoid')(out_pre_16)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_1,out_2,out_3,out_4,out_5,
                                            out_6,out_7,out_8,out_9,out_10,
                                            out_11,out_12,out_13,out_14,out_15, out_16])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f])
    return model



def ModelOneVsRest_SharedPrivate(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared=Dense(nb_n1, activation='sigmoid')(inputs)
    
    

    
    ind_1=concatenate([inputs,x_shared])
    ind_2=concatenate([inputs,x_shared])
    ind_3=concatenate([inputs,x_shared])
    ind_4=concatenate([inputs,x_shared])
    ind_5=concatenate([inputs,x_shared])
    ind_6=concatenate([inputs,x_shared])
    ind_7=concatenate([inputs,x_shared])
    ind_8=concatenate([inputs,x_shared])
    ind_9=concatenate([inputs,x_shared])
    ind_10=concatenate([inputs,x_shared])
    ind_11=concatenate([inputs,x_shared])
    ind_12=concatenate([inputs,x_shared])
    ind_13=concatenate([inputs,x_shared])
    ind_14=concatenate([inputs,x_shared])
    ind_15=concatenate([inputs,x_shared])
    ind_16=concatenate([inputs,x_shared])
    
    x_1=Dense(nb_n2, activation='sigmoid')(ind_1)
    x_2=Dense(nb_n2, activation='sigmoid')(ind_2)
    x_3=Dense(nb_n2, activation='sigmoid')(ind_3)
    x_4=Dense(nb_n2, activation='sigmoid')(ind_4)
    x_5=Dense(nb_n2, activation='sigmoid')(ind_5)
    x_6=Dense(nb_n2, activation='sigmoid')(ind_6)
    x_7=Dense(nb_n2, activation='sigmoid')(ind_7)
    x_8=Dense(nb_n2, activation='sigmoid')(ind_8)
    x_9=Dense(nb_n2, activation='sigmoid')(ind_9)
    x_10=Dense(nb_n2, activation='sigmoid')(ind_10)
    x_11=Dense(nb_n2, activation='sigmoid')(ind_11)
    x_12=Dense(nb_n2, activation='sigmoid')(ind_12)
    x_13=Dense(nb_n2, activation='sigmoid')(ind_13)
    x_14=Dense(nb_n2, activation='sigmoid')(ind_14)
    x_15=Dense(nb_n2, activation='sigmoid')(ind_15)
    x_16=Dense(nb_n2, activation='sigmoid')(ind_16)
    
    out_1=Dense(1, activation='sigmoid')(x_1)
    out_2=Dense(1, activation='sigmoid')(x_2)
    out_3=Dense(1, activation='sigmoid')(x_3)
    out_4=Dense(1, activation='sigmoid')(x_4)
    out_5=Dense(1, activation='sigmoid')(x_5)
    out_6=Dense(1, activation='sigmoid')(x_6)
    out_7=Dense(1, activation='sigmoid')(x_7)
    out_8=Dense(1, activation='sigmoid')(x_8)
    out_9=Dense(1, activation='sigmoid')(x_9)
    out_10=Dense(1, activation='sigmoid')(x_10)
    out_11=Dense(1, activation='sigmoid')(x_11)
    out_12=Dense(1, activation='sigmoid')(x_12)
    out_13=Dense(1, activation='sigmoid')(x_13)
    out_14=Dense(1, activation='sigmoid')(x_14)
    out_15=Dense(1, activation='sigmoid')(x_15)
    out_16=Dense(1, activation='sigmoid')(x_16)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_1,out_2,out_3,out_4,out_5,
                                            out_6,out_7,out_8,out_9,out_10,
                                            out_11,out_12,out_13,out_14,out_15, out_16])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f])
    return model






def ModelOneVsRest_AllShared(nb_n1=100,nb_n2=50):
    #l'apprentissage de chaque partie est independant, c'est du vrai multi task
    inputs = Input(shape=(input_dime,))
    
    x_shared1=Dense(nb_n1, activation='sigmoid')(inputs)
   
    x_1=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_2=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_3=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_4=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_5=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_6=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_7=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_8=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_9=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_10=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_11=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_12=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_13=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_14=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_15=Dense(nb_n2, activation='sigmoid')(x_shared1)
    x_16=Dense(nb_n2, activation='sigmoid')(x_shared1)
    
    out_1=Dense(1, activation='sigmoid')(x_1)
    out_2=Dense(1, activation='sigmoid')(x_2)
    out_3=Dense(1, activation='sigmoid')(x_3)
    out_4=Dense(1, activation='sigmoid')(x_4)
    out_5=Dense(1, activation='sigmoid')(x_5)
    out_6=Dense(1, activation='sigmoid')(x_6)
    out_7=Dense(1, activation='sigmoid')(x_7)
    out_8=Dense(1, activation='sigmoid')(x_8)
    out_9=Dense(1, activation='sigmoid')(x_9)
    out_10=Dense(1, activation='sigmoid')(x_10)
    out_11=Dense(1, activation='sigmoid')(x_11)
    out_12=Dense(1, activation='sigmoid')(x_12)
    out_13=Dense(1, activation='sigmoid')(x_13)
    out_14=Dense(1, activation='sigmoid')(x_14)
    out_15=Dense(1, activation='sigmoid')(x_15)
    out_16=Dense(1, activation='sigmoid')(x_16)
    

    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[out_1,out_2,out_3,out_4,out_5,
                                            out_6,out_7,out_8,out_9,out_10,
                                            out_11,out_12,out_13,out_14,out_15, out_16])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f,loss_f])
    return model







