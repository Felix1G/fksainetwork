impl :: bincode :: Encode for Network
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) -> core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(& self.layers, encoder) ? ; :: bincode
        :: Encode :: encode(& self.has_hidden, encoder) ? ; :: bincode ::
        Encode :: encode(& self.err_terms, encoder) ? ; Ok(())
    }
}