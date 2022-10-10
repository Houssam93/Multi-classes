def hier(nb1,nb2):
    inputs=Input(shape=(512,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g3=Dense(1,activation='sigmoid')(first_layer)
    
    c1=Dense(nb2,activation='sigmoid')(first_layer)
    
    g1=Dense(1,activation='sigmoid')(c1)
    
    c2=Dense(nb2,activation='sigmoid')(c1)
    
    g2=Dense(1,activation='sigmoid')(c2)
    
    c3=Dense(nb2,activation='sigmoid')(c2)
    
    g0=Dense(1,activation='sigmoid')(c3)
    
    g4=Dense(1,activation='sigmoid')(c3)
    
    
    loss_f='binary_crossentropy'
    
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    
    return model