import numpy as np

class DataObject:
    def __init__(self, x_dim, y_dim, I_dim, u_dim=0):
        '''
        Use u_dim=0 to represent single state, non-zero u_dim indicates state at time t, u and state at time t+1
        '''
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.I_dim = I_dim
        self.u_dim = u_dim

        self.y_end = self.x_dim+self.y_dim
        self.I_end = self.y_end+self.I_dim

        self.u_end = None
        self.xn_end = None
        self.yn_end = None
        if self.__is_sequence():
            self.u_end = self.I_end+self.u_dim
            self.xn_end = self.u_end+self.x_dim
            self.yn_end = self.xn_end+self.y_dim

        self.x = None
        self.y = None
        self.I = None
        self.u = None
        self.xn = None
        self.yn = None
        self.In = None
        self.rawData = None

    def fill_from_array(self, input):
        '''
        Used when constructing objects from deserializing array from a file.
        input is a batch.
        '''
        self.rawData = input
        self.x = input[:, 0:self.x_dim]
        self.y = input[:, self.x_dim:self.y_end]
        self.I = input[:, self.y_end:self.I_end]
        
        if self.__is_sequence():
            self.u = input[:, self.I_end:self.u_end]
            self.xn = input[:, self.u_end:self.xn_end]
            self.yn = input[:, self.xn_end:self.yn_end]
            self.In = input[:, self.yn_end:]
    
    def to_vector(self):
        '''
        Used to generate a numpy array row when writing data.
        '''
        res = np.concatenate((self.x, self.y, self.I), axis=0)
        if self.__is_sequence():
            res = np.concatenate((res, self.u, self.xn, self.yn, self.In), axis=0)
        return res

    def from_values(x, y, I):
        '''
        Static method to build the object from Visual Environment implementation.
        '''
        val = DataObject(x.shape[0], y.shape[0], I.shape[0])
        val.x = x
        val.y = y
        val.I = I
        val.rawData = val.to_vector()
        return val

    def from_values_sequence(x, y, I, u, xn, yn, In):
        '''
        Static method to build the object from Visual Environment implementation.
        '''
        val = DataObject(x.shape[0], y.shape[0], I.shape[0], u.shape[0])
        val.x = x
        val.y = y
        val.I = I
        val.u = u
        val.xn = xn
        val.yn = yn
        val.In = In
        val.rawData = val.to_vector()
        return val
    
    def __is_sequence(self):
        return self.u_dim > 0