module Sort where

-- | Sort with pivot
--
-- >>> qsort [10,1,5,8,3,2]
-- [1,2,3,5,8,10]
--
-- prop> qsort xs == Data.List.sort xs
qsort :: [Int] -> [Int]
qsort []     = []
qsort (x:xs) = qsort smaller ++ [x] ++ qsort larger
  where
    smaller = filter (<= x) xs
    larger  = filter (> x) xs
