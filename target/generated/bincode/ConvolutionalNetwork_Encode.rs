impl :: bincode :: Encode for ConvolutionalNetwork
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) -> core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(& self.network, encoder) ? ; :: bincode
        :: Encode :: encode(& self.network_input_arr, encoder) ? ; :: bincode
        :: Encode :: encode(& self.layers, encoder) ? ; :: bincode :: Encode
        :: encode(& self.width, encoder) ? ; :: bincode :: Encode ::
        encode(& self.height, encoder) ? ; :: bincode :: Encode ::
        encode(& self.channels, encoder) ? ; Ok(())
    }
}