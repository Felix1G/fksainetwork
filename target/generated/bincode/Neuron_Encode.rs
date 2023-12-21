impl :: bincode :: Encode for Neuron
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) -> core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(& self.weights_temp, encoder) ? ; ::
        bincode :: Encode :: encode(& self.weights, encoder) ? ; :: bincode ::
        Encode :: encode(& self.bias_temp, encoder) ? ; :: bincode :: Encode
        :: encode(& self.bias, encoder) ? ; :: bincode :: Encode ::
        encode(& self.value, encoder) ? ; :: bincode :: Encode ::
        encode(& self.result, encoder) ? ; :: bincode :: Encode ::
        encode(& self.activation, encoder) ? ; :: bincode :: Encode ::
        encode(& self.error_term, encoder) ? ; Ok(())
    }
}