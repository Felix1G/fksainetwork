impl :: bincode :: Decode for ConvolutionalLayer
{
    fn decode < __D : :: bincode :: de :: Decoder > (decoder : & mut __D) ->
    core :: result :: Result < Self, :: bincode :: error :: DecodeError >
    {
        Ok(Self
        {
            kernels_temp : :: bincode :: Decode :: decode(decoder) ?,
            kernels_layers : :: bincode :: Decode :: decode(decoder) ?,
            kernels : :: bincode :: Decode :: decode(decoder) ?, kernel_size :
            :: bincode :: Decode :: decode(decoder) ?, pooling_size : ::
            bincode :: Decode :: decode(decoder) ?, pooling_method : ::
            bincode :: Decode :: decode(decoder) ?, bias_temp : :: bincode ::
            Decode :: decode(decoder) ?, bias : :: bincode :: Decode ::
            decode(decoder) ?, value : :: bincode :: Decode :: decode(decoder)
            ?, result : :: bincode :: Decode :: decode(decoder) ?, pooled : ::
            bincode :: Decode :: decode(decoder) ?, activation : :: bincode ::
            Decode :: decode(decoder) ?, temp_matrix : :: bincode :: Decode ::
            decode(decoder) ?, temp_pooling_arr : :: bincode :: Decode ::
            decode(decoder) ?,
        })
    }
} impl < '__de > :: bincode :: BorrowDecode < '__de > for ConvolutionalLayer
{
    fn borrow_decode < __D : :: bincode :: de :: BorrowDecoder < '__de > >
    (decoder : & mut __D) -> core :: result :: Result < Self, :: bincode ::
    error :: DecodeError >
    {
        Ok(Self
        {
            kernels_temp : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, kernels_layers : :: bincode ::
            BorrowDecode :: borrow_decode(decoder) ?, kernels : :: bincode ::
            BorrowDecode :: borrow_decode(decoder) ?, kernel_size : :: bincode
            :: BorrowDecode :: borrow_decode(decoder) ?, pooling_size : ::
            bincode :: BorrowDecode :: borrow_decode(decoder) ?,
            pooling_method : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, bias_temp : :: bincode :: BorrowDecode
            :: borrow_decode(decoder) ?, bias : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, value : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, result : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, pooled : :: bincode :: BorrowDecode ::
            borrow_decode(decoder) ?, activation : :: bincode :: BorrowDecode
            :: borrow_decode(decoder) ?, temp_matrix : :: bincode ::
            BorrowDecode :: borrow_decode(decoder) ?, temp_pooling_arr : ::
            bincode :: BorrowDecode :: borrow_decode(decoder) ?,
        })
    }
}