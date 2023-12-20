impl :: bincode :: Decode for ConvolutionalNetwork
{
    fn decode < __D : :: bincode :: de :: Decoder > (decoder : & mut __D) ->
    core :: result :: Result < Self, :: bincode :: error :: DecodeError >
    {
        Ok(Self
        {
            network : :: bincode :: Decode :: decode(decoder) ?,
            network_input_arr : :: bincode :: Decode :: decode(decoder) ?,
            layers : :: bincode :: Decode :: decode(decoder) ?, width : ::
            bincode :: Decode :: decode(decoder) ?, height : :: bincode ::
            Decode :: decode(decoder) ?, channels : :: bincode :: Decode ::
            decode(decoder) ?,
        })
    }
} impl < '__de > :: bincode :: BorrowDecode < '__de > for ConvolutionalNetwork
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de > >
    (decoder : & mut __D) -> core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        Ok(Self
        {
            network : :: bincode :: BorrowDecode :: borrow_decode(decoder) ?,
            network_input_arr : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, layers : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, width : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, height : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, channels : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?,
        })
    }
}