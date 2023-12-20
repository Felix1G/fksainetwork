impl :: bincode :: Encode for ConvolutionalLayer
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) -> core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(& self.kernels_temp, encoder) ? ; ::
        bincode :: Encode :: encode(& self.kernels_layers, encoder) ? ; ::
        bincode :: Encode :: encode(& self.kernels, encoder) ? ; :: bincode ::
        Encode :: encode(& self.kernel_size, encoder) ? ; :: bincode :: Encode
        :: encode(& self.pooling_size, encoder) ? ; :: bincode :: Encode ::
        encode(& self.pooling_method, encoder) ? ; :: bincode :: Encode ::
        encode(& self.bias_temp, encoder) ? ; :: bincode :: Encode ::
        encode(& self.bias, encoder) ? ; :: bincode :: Encode ::
        encode(& self.value, encoder) ? ; :: bincode :: Encode ::
        encode(& self.result, encoder) ? ; :: bincode :: Encode ::
        encode(& self.pooled, encoder) ? ; :: bincode :: Encode ::
        encode(& self.activation, encoder) ? ; :: bincode :: Encode ::
        encode(& self.temp_matrix, encoder) ? ; :: bincode :: Encode ::
        encode(& self.temp_pooling_arr, encoder) ? ; Ok(())
    }
}