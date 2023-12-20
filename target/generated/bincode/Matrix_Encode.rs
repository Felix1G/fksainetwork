impl :: bincode :: Encode for Matrix
{
    fn encode < __E : :: bincode :: enc :: Encoder >
    (& self, encoder : & mut __E) -> core :: result :: Result < (), :: bincode
    :: error :: EncodeError >
    {
        :: bincode :: Encode :: encode(& self.w, encoder) ? ; :: bincode ::
        Encode :: encode(& self.h, encoder) ? ; :: bincode :: Encode ::
        encode(& self.values, encoder) ? ; Ok(())
    }
}