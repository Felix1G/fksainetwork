impl :: bincode :: Decode for Matrix
{
    fn decode < __D : :: bincode :: de :: Decoder > (decoder : & mut __D) ->
    core :: result :: Result < Self, :: bincode :: error :: DecodeError >
    {
        Ok(Self
        {
            w : :: bincode :: Decode :: decode(decoder) ?, h : :: bincode ::
            Decode :: decode(decoder) ?, values : :: bincode :: Decode ::
            decode(decoder) ?,
        })
    }
} impl < '__de > :: bincode :: BorrowDecode < '__de > for Matrix
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de > >
    (decoder : & mut __D) -> core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        Ok(Self
        {
            w : :: bincode :: BorrowDecode :: borrow_decode(decoder) ?, h : ::
            bincode :: BorrowDecode :: borrow_decode(decoder) ?, values : ::
            bincode :: BorrowDecode :: borrow_decode(decoder) ?,
        })
    }
}