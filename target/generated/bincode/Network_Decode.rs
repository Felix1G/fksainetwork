impl :: bincode :: Decode for Network
{
    fn decode < __D : :: bincode :: de :: Decoder > (decoder : & mut __D) ->
    core :: result :: Result < Self, :: bincode :: error :: DecodeError >
    {
        Ok(Self
        {
            layers : :: bincode :: Decode :: decode(decoder) ?, has_hidden :
            :: bincode :: Decode :: decode(decoder) ?, err_terms : :: bincode
            :: Decode :: decode(decoder) ?,
        })
    }
} impl < '__de > :: bincode :: BorrowDecode < '__de > for Network
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de > >
    (decoder : & mut __D) -> core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        Ok(Self
        {
            layers : :: bincode :: BorrowDecode :: borrow_decode(decoder) ?,
            has_hidden : :: bincode :: BorrowDecode :: borrow_decode(decoder)
            ?, err_terms : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?,
        })
    }
}