-- |
--
-- >>> qsort [10,1,5,8,3,2]
-- [1,2,3,5,8,10]
qsort :: [Int] -> [Int]
qsort []     = []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
  where
    smaller = [a | a <- xs, a <= x]
    larger  = [b | b <- xs, b > x]
