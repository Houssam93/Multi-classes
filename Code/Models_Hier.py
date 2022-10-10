def hier(nb1,nb2):
    inputs=Input(shape=(600,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g2=Dense(1,activation='sigmoid')(first_layer)
    
    g3=Dense(1,activation='sigmoid')(first_layer)
    
    c1=Dense(nb2,activation='sigmoid')(first_layer)
    
    g0=Dense(1,activation='sigmoid')(c1)
    
    c2=Dense(nb2,activation='sigmoid')(c1)

    g1=Dense(1,activation='sigmoid')(c2)
    
    g4=Dense(1,activation='sigmoid')(c2)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model

def hier_connected(nb1,nb2):
    inputs=Input(shape=(600,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g2=Dense(1,activation='sigmoid')(first_layer)
    
    g3=Dense(1,activation='sigmoid')(first_layer)
    
    inputs_c1=concatenate([first_layer,g2,g3])
    
    c1=Dense(nb2,activation='sigmoid')(inputs_c1)
    
    g0=Dense(1,activation='sigmoid')(c1)
    
    inputs_c2=concatenate([c1,g0])
    
    c2=Dense(nb2,activation='sigmoid')(inputs_c2)
    
    g1=Dense(1,activation='sigmoid')(c2)
    
    g4=Dense(1,activation='sigmoid')(c2)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model

def hier_fully_connected(nb1,nb2):
    inputs=Input(shape=(600,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g2=Dense(1,activation='sigmoid')(first_layer)
    
    g3=Dense(1,activation='sigmoid')(first_layer)
    
    inputs_c1=concatenate([first_layer,g2,g3])
    
    c1=Dense(nb2,activation='sigmoid')(inputs_c1)
    
    g0=Dense(1,activation='sigmoid')(c1)
    
    inputs_c2=concatenate([c1,g2,g3,g0])
    
    c2=Dense(nb2,activation='sigmoid')(inputs_c2)

    g1=Dense(1,activation='sigmoid')(c2)
    
    g4=Dense(1,activation='sigmoid')(c2)
    
    
    loss_f='binary_crossentropy'
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    return model


def hier_coga(nb1,nb2):
    inputs=Input(shape=(600,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g1=Dense(1,activation='sigmoid')(first_layer)
    
    g0=Dense(1,activation='sigmoid')(first_layer)
    
    g2=Dense(1,activation='sigmoid')(first_layer)
    
    g4=Dense(1,activation='sigmoid')(first_layer)
    
    c1=Dense(nb2,activation='sigmoid')(first_layer)
    
    g3=Dense(1,activation='sigmoid')(c1)
    
    loss_f='binary_crossentropy'
    
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])

    return model

def hier_connected_coga(nb1,nb2):    
    
    inputs=Input(shape=(600,))
    
    first_layer=Dense(nb1,activation='sigmoid')(inputs)
    
    g1=Dense(1,activation='sigmoid')(first_layer)
    
    g0=Dense(1,activation='sigmoid')(first_layer)
    
    g2=Dense(1,activation='sigmoid')(first_layer)
    
    g4=Dense(1,activation='sigmoid')(first_layer)
    
    inputs_c1=concatenate([first_layer,g1,g0,g2,g4])
    
    c1=Dense(nb2,activation='sigmoid')(inputs_c1)
    
    g3=Dense(1,activation='sigmoid')(c1)
    
    
    loss_f='binary_crossentropy'
    
    model = Model(inputs=[inputs], outputs=[g0,g1,g2,g3,g4])
    model.compile(optimizer='Adam', loss= [loss_f,loss_f,loss_f,loss_f,loss_f])
    
    return model