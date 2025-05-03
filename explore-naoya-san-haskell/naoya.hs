-- This code is copied from https://atcoder.jp/contests/abc402/submissions/64997179
-- {{{ Imports and Language Extensions
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# OPTIONS_GHC -Wno-unused-imports -Wno-unused-top-binds -Wno-dodgy-imports -Wno-orphans -Wno-unrecognised-pragmas #-}

import Control.Applicative
import Control.Exception
import Control.Monad
import Control.Monad.Extra hiding (loop)
import Control.Monad.Primitive (PrimMonad, PrimState)
import Control.Monad.ST
import Control.Monad.State
import Data.Array.Base (IArray (numElements))
import Data.Array.IArray
import Data.Array.IO
import Data.Array.MArray
import Data.Array.ST
import Data.Array.Unboxed (UArray)
import Data.Bifunctor
import Data.Bits
import Data.Bool
import Data.ByteString.Char8 qualified as BS
import Data.ByteString.Lazy.Char8 qualified as BSL
import Data.Char
import Data.Coerce (coerce)
import Data.Foldable
import Data.Function
import Data.HashMap.Strict qualified as HM
import Data.HashPSQ qualified as HashPSQ
import Data.HashSet qualified as HS
import Data.Hashable qualified as Hashable
import Data.Heap qualified as Heap
import Data.IORef
import Data.IntMap.Strict qualified as IM
import Data.IntPSQ qualified as IntPSQ
import Data.IntSet qualified as IS
import Data.List
import Data.List.Extra hiding ((!?))
import Data.Map.Strict qualified as Map
import Data.Maybe
import Data.Ord hiding (clamp)
import Data.STRef
import Data.Sequence (Seq (Empty, (:<|), (:|>)), (<|), (|>))
import Data.Sequence qualified as Seq
import Data.Set qualified as Set
import Data.Text qualified as Text
import Data.Text.IO qualified as TIO
import Data.Traversable
import Data.Tuple
import Data.Tuple.Extra (first3, fst3, second3, snd3, thd3, third3)
import Data.Vector qualified as V
import Data.Vector.Algorithms.Intro qualified as VAI
import Data.Vector.Generic qualified as VG
import Data.Vector.Unboxed qualified as VU
import Data.Vector.Unboxed.Mutable qualified as VUM
import Debug.Trace (traceShow)
import GHC.IO (unsafePerformIO)
import System.Environment (lookupEnv)
import System.Exit (exitSuccess)
import System.IO (hFlush, stdout)
import System.Random (newStdGen, randoms)

-- }}}

main :: IO ()
main = do
  s <- getLine

  putStrLn [c | c <- s, isUpper c]

-- {{{ My Prelude

{-- Constants --}

modulus :: Int
-- modulus = 1000000007
modulus = 998244353

factCacheSize :: Int
factCacheSize = 2 * 10 ^ (6 :: Int)

{-- IntMod --}

addMod :: Int -> Int -> Int
addMod x y = (x + y) `mod` modulus
{-# INLINE addMod #-}

subMod :: Int -> Int -> Int
subMod x y = (x - y) `mod` modulus
{-# INLINE subMod #-}

mulMod :: Int -> Int -> Int
mulMod x y = (x * y) `mod` modulus
{-# INLINE mulMod #-}

sumMod :: [Int] -> Int
sumMod = foldl' addMod 0
{-# INLINE sumMod #-}

productMod :: [Int] -> Int
productMod = foldl' mulMod 1
{-# INLINE productMod #-}

newtype IntMod = IntMod Int deriving (Eq, Show, Read)

instance Num IntMod where
  IntMod x + IntMod y = IntMod (addMod x y)
  IntMod x - IntMod y = IntMod (subMod x y)
  IntMod x * IntMod y = IntMod (mulMod x y)
  fromInteger n = IntMod (fromInteger (n `mod` fromIntegral @Int modulus))
  abs = undefined
  signum = undefined

toIntMod :: Int -> IntMod
toIntMod a = IntMod (a `mod` modulus)

exEuclid :: Int -> Int -> (Int, Int, Int)
exEuclid = fn 1 0 0 1
  where
    fn s t s' t' a b
      | b == 0 = (a, s, t)
      | otherwise =
          let (q, r) = a `divMod` b
           in fn s' t' (s - q * s') (t - q * t') b r

invMod :: Int -> Int
invMod a = case exEuclid a modulus of
  (1, s, _) -> s `mod` modulus
  (-1, s, _) -> (-s) `mod` modulus
  _anyOfFailure -> error $ show a ++ " has no inverse modulo" ++ show @Int modulus

-- 逆元 IntMod / IntMod に掛けるだけで OK
invIntMod :: IntMod -> IntMod
invIntMod (IntMod a) = IntMod (invMod a)

-- 計算量は r のサイズにしか依存してない. n が 10^9 とか大きくても計算できる
ncrMod :: Int -> Int -> IntMod
ncrMod n r = a * invIntMod b
  where
    a = product $ map IntMod (take r [n, n - 1 ..])
    b = product $ map IntMod (take r [r, r - 1 ..])

{-- combMod --}

combMod :: Int -> Int -> IntMod
combMod n k
  | n < k = 0
  | n < 0 || k < 0 = 0
  | otherwise = fact n * (factInv k * factInv (n - k))
{-# INLINE combMod #-}

initComb :: Int -> (UArray Int Int, UArray Int Int)
initComb size = runST $ do
  fact_ <- newArray (0, size + 5) 0 :: ST s (STUArray s Int Int)
  factInv_ <- newArray (0, size + 5) 0 :: ST s (STUArray s Int Int)
  inv <- newArray (0, size + 5) 0 :: ST s (STUArray s Int Int)

  writeArray fact_ 0 1
  writeArray fact_ 1 1
  writeArray factInv_ 0 1
  writeArray factInv_ 1 1
  writeArray inv 1 1

  for_ [2 .. size + 5] $ \i -> do
    !x <- readArray fact_ (i - 1)
    writeArray fact_ i $ x `mulMod` i

    !y <- readArray inv (modulus `mod` i)
    writeArray inv i $ modulus `subMod` y `mulMod` (modulus `div` i)

    !z <- readArray factInv_ (i - 1)
    !z' <- readArray inv i
    writeArray factInv_ i $ z `mulMod` z'

  fact' <- freeze fact_
  factInv' <- freeze factInv_
  return (fact', factInv')

factCache :: (UArray Int Int, UArray Int Int)
factCache = initComb factCacheSize
{-# NOINLINE factCache #-}

fact :: Int -> IntMod
fact n = IntMod (fst factCache ! n)
{-# INLINE fact #-}

factInv :: Int -> IntMod
factInv n = IntMod (snd factCache ! n)
{-# INLINE factInv #-}

{-- IO --}

yn :: Bool -> String
yn = bool "No" "Yes"

printYn :: Bool -> IO ()
printYn = putStrLn . yn

printTuple :: (Show a) => (a, a) -> IO ()
printTuple (i, j) = printList [i, j]

printList :: (Show a) => [a] -> IO ()
printList = putStrLn . unwords . map show

printFlush :: (Show a) => a -> IO ()
printFlush x = print x >> hFlush stdout

printListFlush :: (Show a) => [a] -> IO ()
printListFlush xs = printList xs >> hFlush stdout

putStrLnFlush :: String -> IO ()
putStrLnFlush s = putStrLn s >> hFlush stdout

getInt :: IO Int
getInt = fst . fromJust . BS.readInt <$> BS.getLine

getInteger :: IO Integer
getInteger = fst . fromJust . BS.readInteger <$> BS.getLine

getInts :: IO [Int]
getInts = unfoldr (BS.readInt . BS.dropWhile isSpace) <$> BS.getLine

getIntegers :: IO [Integer]
getIntegers = unfoldr (BS.readInteger . BS.dropWhile isSpace) <$> BS.getLine

getTuple :: IO (Int, Int)
getTuple = auto

getTuple3 :: IO (Int, Int, Int)
getTuple3 = auto

getWeightedEdge :: IO ((Int, Int), Int)
getWeightedEdge = do
  [u, v, w] <- getInts
  return ((u, v), w)

getIntArray :: (Int, Int) -> IO (UArray Int Int)
getIntArray b = listArray @UArray b <$> getInts

getCharArray :: (Int, Int) -> IO (UArray Int Char)
getCharArray b = listArray @UArray b <$> getLine

getStringArray :: IO (UArray Int Char)
getStringArray = do
  s <- BS.getLine
  return $ listArray @UArray (1, BS.length s) (BS.unpack s)

getCharGrid :: ((Int, Int), (Int, Int)) -> IO (UArray (Int, Int) Char)
getCharGrid b@((s, _), (h, _)) = do
  xs <- replicateM (h + 1 - s) BS.getLine
  return (listArray @UArray b $ BS.unpack $ BS.concat xs)

getIntGrid :: ((Int, Int), (Int, Int)) -> IO (UArray (Int, Int) Int)
getIntGrid b@((s, _), (h, _)) = do
  xs <- replicateM (h + 1 - s) getInts
  return (listArray @UArray b $ concat xs)

printGrid :: (Applicative f, IArray a i, Ix v) => ([i] -> f b) -> a (v, Int) i -> f ()
printGrid f grid = traverse_ f $ chunksOf w (elems grid)
  where
    ((_, w1), (_, w2)) = bounds grid
    w = w2 + 1 - w1

printCharGrid :: (IArray a Char, Ix v) => a (v, Int) Char -> IO ()
printCharGrid = printGrid putStrLn

printIntGrid :: (Show e, IArray a e, Ix v) => a (v, Int) e -> IO ()
printIntGrid = printGrid printList

{-- auto --}
-- via https://github.com/toyboot4e/toy-lib/blob/main/src/ToyLib/IO.hs

-- | Read from a space-delimited `ByteString`.
class ReadBS a where
  {-# INLINE convertBS #-}
  convertBS :: BS.ByteString -> a
  default convertBS :: (Read a) => BS.ByteString -> a
  convertBS = read . BS.unpack

  -- | For use with `U.unfoldrExactN`.
  {-# INLINE readBS #-}
  readBS :: BS.ByteString -> (a, BS.ByteString)
  readBS !bs =
    let (!bs1, !bs2) = BS.break isSpace bs
     in (convertBS bs1, bs2)

  -- | For use with `U.unfoldr`.
  {-# INLINE readMayBS #-}
  readMayBS :: BS.ByteString -> Maybe (a, BS.ByteString)
  readMayBS !bs
    | BS.null bs = Nothing
    | otherwise =
        let (!bs1, !bs2) = BS.break isSpace bs
         in Just (convertBS bs1, bs2)

instance ReadBS Int where
  {-# INLINE convertBS #-}
  convertBS = fst . readBS
  {-# INLINE readBS #-}
  readBS = fromJust . readMayBS
  {-# INLINE readMayBS #-}
  readMayBS = BS.readInt

instance ReadBS Integer where
  {-# INLINE convertBS #-}
  convertBS = fst . readBS
  {-# INLINE readBS #-}
  readBS = fromJust . readMayBS
  {-# INLINE readMayBS #-}
  readMayBS = BS.readInteger

instance ReadBS IntMod where
  {-# INLINE convertBS #-}
  convertBS = toIntMod . fst . readBS
  {-# INLINE readBS #-}
  readBS = first toIntMod . fromJust . readMayBS
  {-# INLINE readMayBS #-}
  readMayBS s = case BS.readInt s of
    Nothing -> Nothing
    Just (x, xs) -> Just (toIntMod x, xs)

instance ReadBS Float

instance ReadBS Double

instance ReadBS Char where
  {-# INLINE convertBS #-}
  convertBS = BS.head

instance ReadBS String where
  {-# INLINE convertBS #-}
  convertBS = BS.unpack

instance ReadBS BS.ByteString where
  {-# INLINE convertBS #-}
  convertBS = id

instance (ReadBS a1, ReadBS a2) => ReadBS (a1, a2) where
  {-# INLINE convertBS #-}
  convertBS !bs0 =
    let (!a1, !bs1) = readBS (BS.dropWhile isSpace bs0)
        !a2 = convertBS (BS.dropWhile isSpace bs1)
     in (a1, a2)
  {-# INLINE readBS #-}
  readBS = fromJust . readMayBS
  {-# INLINE readMayBS #-}
  readMayBS !bs0 = do
    (!x1, !bs1) <- readMayBS bs0
    (!x2, !bs2) <- readMayBS bs1
    Just ((x1, x2), bs2)

instance (ReadBS a1, ReadBS a2, ReadBS a3) => ReadBS (a1, a2, a3) where
  {-# INLINE convertBS #-}
  convertBS !bs0 =
    let (!a1, !bs1) = readBS (BS.dropWhile isSpace bs0)
        (!a2, !bs2) = readBS (BS.dropWhile isSpace bs1)
        !a3 = convertBS (BS.dropWhile isSpace bs2)
     in (a1, a2, a3)
  {-# INLINE readBS #-}
  readBS = fromJust . readMayBS
  {-# INLINE readMayBS #-}
  readMayBS !bs0 = do
    (!x1, !bs1) <- readMayBS bs0
    (!x2, !bs2) <- readMayBS bs1
    (!x3, !bs3) <- readMayBS bs2
    Just ((x1, x2, x3), bs3)

-- instance (ReadBS a1, ReadBS a2, ReadBS a3, ReadBS a4) => ReadBS (a1, a2, a3, a4) where
--   {-# INLINE convertBS #-}
--   convertBS !bs0 =
--     let (!a1, !bs1) = readBS (BS.dropWhile isSpace bs0)
--         (!a2, !bs2) = readBS (BS.dropWhile isSpace bs1)
--         (!a3, !bs3) = readBS (BS.dropWhile isSpace bs2)
--         !a4 = convertBS (BS.dropWhile isSpace bs3)
--      in (a1, a2, a3, a4)
--   {-# INLINE readBS #-}
--   readBS = fromJust . readMayBS
--   {-# INLINE readMayBS #-}
--   readMayBS !bs0 = do
--     (!x1, !bs1) <- readMayBS bs0
--     (!x2, !bs2) <- readMayBS bs1
--     (!x3, !bs3) <- readMayBS bs2
--     (!x4, !bs4) <- readMayBS bs3
--     Just ((x1, x2, x3, x4), bs4)

-- | Parses one line via the `ReadBS` class.
auto :: (ReadBS a) => IO a
auto = convertBS <$> BS.getLine

{-- Control --}

foldFor' :: (Foldable t) => b -> t a -> (b -> a -> b) -> b
foldFor' initial xs f = foldl' f initial xs

foldForM :: (Foldable t, Monad m) => b -> t a -> (b -> a -> m b) -> m b
foldForM initial xs m = foldM m initial xs

foldForM_ :: (Foldable t, Monad m) => b -> t a -> (b -> a -> m b) -> m ()
foldForM_ initial xs m = foldM_ m initial xs

mapFor :: [a] -> (a -> b) -> [b]
mapFor xs f = map f xs

mapAccumLFor :: (Traversable t) => a -> t b -> (a -> b -> (a, c)) -> (a, t c)
mapAccumLFor initial xs f = mapAccumL f initial xs

mapAccumMFor :: (Monad m, Traversable t) => s -> t a -> (s -> a -> m (s, b)) -> m (s, t b)
mapAccumMFor initial xs f = mapAccumM f initial xs

times :: Int -> (a -> a) -> a -> a
times n f s0 = snd $ until ((== n) . fst) (bimap succ f) (0 :: Int, s0)

scanM :: (Monad m) => (a -> b -> m a) -> a -> [b] -> m [a]
scanM _ acc [] = return [acc]
scanM f acc (x : xs) = do
  next <- f acc x
  rest <- scanM f next xs
  return (acc : rest)

scanForM :: (Monad m) => a -> [b] -> (a -> b -> m a) -> m [a]
scanForM initial xs m = scanM m initial xs

mapAccumM :: (Monad m, Traversable t) => (s -> a -> m (s, b)) -> s -> t a -> m (s, t b)
mapAccumM f s t = do
  (t', s') <-
    runStateT
      ( traverse
          ( \a ->
              StateT
                ( \s'' -> do
                    (s''', b) <- f s'' a
                    return (b, s''')
                )
          )
          t
      )
      s
  return (s', t')

{-- Ord --}

clamp :: (Ord a) => (a, a) -> a -> a
clamp (low, high) a = min high (max a low)

{-- List --}

indexed :: Int -> [a] -> [(Int, a)]
indexed i0 = zip [i0 :: Int ..]
{-# INLINE indexed #-}

isUnique :: (Ord a) => [a] -> Bool
isUnique xs = length xs == length (nubOrd xs)

countBy :: (Foldable t) => (e -> Bool) -> t e -> Int
countBy predicate = foldl' (\acc a -> if predicate a then acc + 1 else acc) 0

-- リストを 前、LR区間、その後ろに3分割する
-- 0-indexed 始まり半開区間で指定する
-- >>>  splitLR 3 13 "merrychristmas"
-- ("mer","rychristma","s")
splitLR :: Int -> Int -> [a] -> ([a], [a], [a])
splitLR l r xs = do
  let (pre, remain) = splitAt l xs
      (mid, post) = splitAt (r - l) remain
  (pre, mid, post)

-- S の i 以降で f を満たすインデックスを探す
-- >>> findIndexFrom 1 (== '0') "01234567890123456789"
-- Just 10
-- >>> findIndexFrom 0 (== '0') "01234567890123456789"
-- Just 0
findIndexFrom :: Int -> (a -> Bool) -> [a] -> Maybe Int
findIndexFrom i f s =
  case findIndex f (drop i s) of
    Just j -> Just (i + j)
    Nothing -> Nothing

elemIndexFrom :: (Eq a) => Int -> a -> [a] -> Maybe Int
elemIndexFrom i x = findIndexFrom i (== x)

-- >>> deleteAt 2 [1, 2, 3, 4]
-- [1,2,4]
deleteAt :: Int -> [a] -> [a]
deleteAt i xs = do
  let (pre, post) = splitAt i xs
  pre ++ tail post

minimumDef :: (Foldable t, Ord p) => p -> t p -> p
minimumDef def xs
  | null xs = def
  | otherwise = minimum xs

maximumDef :: (Foldable t, Ord p) => p -> t p -> p
maximumDef def xs
  | null xs = def
  | otherwise = maximum xs

-- groupByAdjacent ··· groupBy と異なり隣の要素同士を比較する
-- >>> groupByAdjacent (<) [1,2,3,2,3,4,2,3,5,10,3,4,5]
-- [[1,2,3],[2,3,4],[2,3,5,10],[3,4,5]]
groupByAdjacent :: (a -> a -> Bool) -> [a] -> [[a]]
groupByAdjacent p = foldr f []
  where
    f x ((y : ys) : zs)
      | p x y = (x : y : ys) : zs
      | otherwise = [x] : ((y : ys) : zs)
    f x [] = [[x]]
    f _ zs = zs

-- リストを指定した n 個の、最低1つは要素が入っているグループに分割
-- >>> groupsN 2 [1 .. 3]
-- [[[2,1],[3]],[[3,1],[2]],[[1],[3,2]]]
groupsN :: Int -> [Int] -> [[[Int]]]
groupsN n = do
  let g = listArray @Array (1, n) $ replicate n []
  dfs_ (g, 1)
  where
    dfs_ (g, k) []
      | n < k = [elems g]
      | otherwise = []
    dfs_ (g, k) (x : xs) = do
      i <- [1 .. min k n]

      let g' = accum (flip (:)) g [(i, x)]

      if k == i
        then dfs_ (g', k + 1) xs
        else dfs_ (g', k) xs

-- グループを BitSet で返却する
-- >>> groupsNBS 2 [1 .. 3]
-- [[fromList [1,2],fromList [3]],[fromList [1,3],fromList [2]],[fromList [1],fromList [2,3]]]
groupsNBS :: Int -> [Int] -> [[BitSet]]
groupsNBS n = do
  let g = listArray @Array (1, n) $ replicate n emptyBS
  dfs_ (g, 1)
  where
    dfs_ (g, k) []
      | n < k = [elems g]
      | otherwise = []
    dfs_ (g, k) (x : xs) = do
      i <- [1 .. min k n]

      let g' = accum (flip insertBS) g [(i, x)]

      if k == i
        then dfs_ (g', k + 1) xs
        else dfs_ (g', k) xs

-- n 個を指定せず全グループ分けパターンを列挙
-- >>> groupPartitions [1, 2, 3]
-- [[[1],[2],[3]],[[1,2],[3]],[[2],[1,3]],[[1],[2,3]],[[1,2,3]]]
groupPartitions :: [a] -> [[[a]]]
groupPartitions [] = [[]]
groupPartitions (x : xs) =
  concatMap f (groupPartitions xs)
  where
    f ps = ([x] : ps) : [let (front, g : back) = splitAt i ps in front ++ ((x : g) : back) | i <- [0 .. length ps - 1]]

-- 和 x を n 個の数で構成するパターンを全列挙する
-- "_____" を ["__", "_", "__"] の 3つに分割するなどに使える
-- >>> sumCombinations 3 5
-- [[1,1,3],[1,2,2],[1,3,1],[2,1,2],[2,2,1],[3,1,1]]
sumCombinations :: (Eq a, Num a, Enum a) => a -> a -> [[a]]
sumCombinations 0 _ = [[]]
sumCombinations 1 !x = [[x]]
sumCombinations !n !x = do
  !a <- [1 .. x - n + 1]
  !as <- sumCombinations (n - 1) (x - a)
  return (a : as)

-- 並び順は維持しつつ分割可能箇所で分割するパターンを全列挙する
--- >>> allPartitions [1, 2, 5]
-- [[[1],[2],[5]],[[1],[2,5]],[[1,2],[5]],[[1,2,5]]]
allPartitions :: [a] -> [[[a]]]
allPartitions as = do
  xs <- subsequences [1 .. length as - 1]

  let ((_, remain), grouped) = mapAccumL f (0, as) xs
        where
          f (i, acc) j =
            let k = j - i
                (pre, post) = splitAt k acc
             in ((j, post), pre)

  [grouped ++ [remain]]

-- 2N 個の要素を N 個の2ペアに分かれる方法を全列挙
-- >>> pairPermutations [1 .. 4]
-- [[(1,2),(3,4)],[(1,3),(2,4)],[(1,4),(2,3)]]
pairPermutations :: (Eq a) => [a] -> [[(a, a)]]
pairPermutations [] = [[]]
pairPermutations (x : xs) = do
  y <- xs
  sub <- pairPermutations (delete y xs)
  [(x, y) : sub]

-- | 重複を生成しない permutations
distinctPermutations :: (Ord a) => [a] -> [[a]]
distinctPermutations vs = permute (length vs) (sort vs)
  where
    permute 0 _ = [[]]
    permute _ [] = []
    permute n xs = [x : ys | (x, xs') <- select xs, ys <- permute (n - 1) xs']

    select :: (Ord a) => [a] -> [(a, [a])]
    select [] = []
    select (x : xs) = (x, xs) : [(y, x : ys) | (y, ys) <- select xs, y /= x]

combinations :: Int -> [a] -> [[a]]
combinations _ [] = []
combinations n as@(_ : xs)
  | n == 0 = [[]]
  | n == 1 = map pure as
  | n == l = pure as
  | n > l = []
  | otherwise = run (l - 1) (n - 1) as $ combinations (n - 1) xs
  where
    l = length as

    run :: Int -> Int -> [a] -> [[a]] -> [[a]]
    run m k ys cs
      | m == k = map (ys ++) cs
      | otherwise = case take (m - k + 1) ys of
          (q : qs) -> do
            let dc = product [(m - k + 1) .. (m - 1)] `div` product [1 .. (k - 1)]
            map (q :) cs ++ run (m - 1) k qs (drop dc cs)
          [] -> error "Invalid Case"

comb2 :: [a] -> [(a, a)]
comb2 = map f . combinations 2
  where
    f [a, b] = (a, b)
    f _ = error "assert"

prevPermutation :: [Int] -> Maybe [Int]
prevPermutation ps = do
  let ps' = reverse ps

  case dropWhile (uncurry (>)) $ zip ps' (tail ps') of
    [] -> Nothing
    (_, x) : _ -> do
      i <- elemIndex x ps
      y <- IS.lookupLT x (IS.fromList (drop (i + 1) ps))
      j <- elemIndex y ps

      let as' = listArray @UArray (0, n - 1) ps // [(j, x), (i, y)]
          (pre, post) = splitAt (i + 1) (elems as')

      return (pre ++ reverse post)
  where
    n = length ps

sorted :: (Ord a) => [a] -> Bool
sorted xs = and $ zipWith (<=) xs (tail xs)

sort' :: (VUM.Unbox a, Ord a) => [a] -> [a]
sort' xs = VU.toList $ VU.modify (VAI.sortBy compare) (VU.fromList xs)

sortBy' :: (VUM.Unbox a) => (a -> a -> Ordering) -> [a] -> [a]
sortBy' f xs = VU.toList $ VU.modify (VAI.sortBy f) (VU.fromList xs)

sortOn' :: (VUM.Unbox a2, VUM.Unbox a1, Ord a2) => (a1 -> a2) -> [a1] -> [a1]
sortOn' f xs =
  VU.toList $
    VU.map snd $
      VU.modify (VAI.sortBy (comparing fst)) $
        VU.map (\x -> let y = f x in y `seq` (y, x)) $
          VU.fromList xs

nubOrd' :: (VUM.Unbox a, Ord a) => [a] -> [a]
nubOrd' = VU.toList . VU.uniq . VU.modify (VAI.sortBy compare) . VU.fromList

{-- Tuple --}

instance (Num a) => Num (a, a) where
  (x1, x2) + (y1, y2) = (x1 + y1, x2 + y2)
  (x1, x2) - (y1, y2) = (x1 - y1, x2 - y2)
  (x1, x2) * (y1, y2) = (x1 * y1, x2 * y2)
  negate (x1, x2) = (negate x1, negate x2)
  abs (x1, x2) = (abs x1, abs x2)
  signum (x1, x2) = (signum x1, signum x2)
  fromInteger n = (fromInteger n, fromInteger n)

instance (Num a) => Num (a, a, a) where
  (x1, x2, x3) + (y1, y2, y3) = (x1 + y1, x2 + y2, x3 + y3)
  (x1, x2, x3) - (y1, y2, y3) = (x1 - y1, x2 - y2, x3 - y3)
  (x1, x2, x3) * (y1, y2, y3) = (x1 * y1, x2 * y2, x3 * y3)
  negate (x1, x2, x3) = (negate x1, negate x2, negate x3)
  abs (x1, x2, x3) = (abs x1, abs x2, abs x3)
  signum (x1, x2, x3) = (signum x1, signum x2, signum x3)
  fromInteger n = (fromInteger n, fromInteger n, fromInteger n)

filterOnFst :: (a -> Bool) -> [(a, b)] -> [(a, b)]
filterOnFst f = filter (f . fst)

filterOnSnd :: (b -> Bool) -> [(a, b)] -> [(a, b)]
filterOnSnd f = filter (f . snd)

findOnFst :: (Foldable t) => (b1 -> Bool) -> t (b1, b2) -> Maybe (b1, b2)
findOnFst f = find (f . fst)

findOnSnd :: (Foldable t) => (b -> Bool) -> t (a, b) -> Maybe (a, b)
findOnSnd f = find (f . snd)

rotateRight :: (Num b) => (b, a) -> (a, b)
rotateRight (h, w) = (w, -h)
{-# INLINE rotateRight #-}

rotateLeft :: (Num a) => (b, a) -> (a, b)
rotateLeft (h, w) = (-w, h)
{-# INLINE rotateLeft #-}

{-- Map / IntMap --}

accumMap :: (Foldable t1, Ord k) => (t2 -> t3 -> t2) -> t2 -> t1 (k, t3) -> Map.Map k t2
accumMap combine def = foldl' f Map.empty
  where
    f acc (key, value) = Map.alter (\x -> Just (combine (fromMaybe def x) value)) key acc

accumIntMap :: (Foldable t1) => (t2 -> t3 -> t2) -> t2 -> t1 (IS.Key, t3) -> IM.IntMap t2
accumIntMap combine def = foldl' f IM.empty
  where
    f acc (key, value) = IM.alter (\x -> Just (combine (fromMaybe def x) value)) key acc

accumHashMap :: (Foldable t1, Hashable.Hashable k, Eq k) => (t2 -> t3 -> t2) -> t2 -> t1 (k, t3) -> HM.HashMap k t2
accumHashMap combine def = foldl' f HM.empty
  where
    f acc (key, value) = HM.alter (\x -> Just (combine (fromMaybe def x) value)) key acc

{-- String --}

stringToInt :: String -> Int
stringToInt = fromDigits 10 . map digitToInt
{-# INLINE stringToInt #-}

-- | 部分文字列検索 / str の中から s にマッチする位置を探しリストで全て返す
-- >>> findString "hello" "hello, hello, world, hello"
-- [0,7,21]
findString :: (Eq a) => [a] -> [a] -> [Int]
findString s str = findIndices (isPrefixOf s) (tails str)

-- | 部分文字列の取得 0-indexed で開始位置を指定、指定した長さの部分文字列を返す
-- >>> substring 0 5 "Hello, World"
-- "Hello"
substring :: Int -> Int -> String -> String
substring start len = take len . drop start

-- | s に含まれる部分文字列をすべて返す
-- >>> substrings "TOYOTA"
-- ["T","TO","TOY","TOYO","TOYOT","TOYOTA","O","OY","OYO","OYOT","OYOTA","Y","YO","YOT","YOTA","O","OT","OTA","T","TA","A"]
substrings :: String -> [String]
substrings s = [substring i l s | i <- [0 .. n], l <- [1 .. n - i]]
  where
    n = length s

-- | s に含まれる長さ k の部分文字列をすべて返す
-- >>> substringsK 3 "TOYOTA"
-- ["TOY","OYO","YOT","OTA"]
substringsK :: Int -> String -> [String]
substringsK k s = [substring i k s | i <- [0 .. length s - k]]

-- ByteString の substring
-- ByteString は BS.drop も BS.take も O(1)
byteSubstring :: Int -> Int -> BS.ByteString -> BS.ByteString
byteSubstring start len = BS.take len . BS.drop start

byteSubstringsK :: Int -> BS.ByteString -> [BS.ByteString]
byteSubstringsK k s = [byteSubstring i k s | i <- [0 .. BS.length s - k]]

-- ByteString 含まれる全ての部分文字列を ByteString のリストで返す
byteSubstrings :: BS.ByteString -> [BS.ByteString]
byteSubstrings s = [byteSubstring i l s | i <- [0 .. n], l <- [1 .. n - i]]
  where
    n = BS.length s

-- >>> pad0 6 5
-- "000005"
pad0 :: Int -> Int -> String
pad0 n x = replicate (n - length x') '0' ++ x'
  where
    x' = show x

{-- Run Length --}

rle :: (Eq a) => [a] -> [(a, Int)]
rle = map (\x -> (head x, length x)) . group

rleOn :: (Eq b) => (a -> b) -> [a] -> [(a, Int)]
rleOn f = map (\g -> (head g, length g)) . groupOn f

rleBS :: BS.ByteString -> [(Char, Int)]
rleBS = map (\x -> (BS.head x, BS.length x)) . BS.group

{-- Math --}

nc2 :: Int -> Int
nc2 n = (n * (n - 1)) `div` 2

ncr :: Int -> Int -> Int
ncr n r = product (take r [n, n - 1 ..]) `div` product (take r [r, r - 1 ..])

ceildiv :: (Integral a) => a -> a -> a
ceildiv a b = (a + b - 1) `div` b

-- >>> log2GE 5
-- 3
log2GE :: (Integral b, Num a, Ord a) => a -> b
log2GE n = until ((>= n) . (2 ^)) succ 0

-- >>> log2LE 5
-- 2
log2LE :: (Integral b, Num a, Ord a) => a -> b
log2LE n = until (\x -> 2 ^ (x + 1) > n) succ 0

-- | 整数 x を d で何回割る事が出来るか
-- >>> divCount 2 12
-- 2
divCount :: (Integral a) => a -> a -> Int
divCount d x
  | d == 0 = error "divisor cannot be zero"
  | x == 0 = maxBound
  | otherwise = fst $ until (\(_, x') -> x' `mod` d /= 0) (bimap succ (`div` d)) (0, x)

-- x の 10^i の位以下を四捨五入する (最右が 0 番目)
-- >>> roundAt 0 123
-- 120
-- >>> roundAt 1 2050
-- 2100
roundAt :: (Integral b, Integral a) => b -> a -> a
roundAt k x
  | m >= 5 = (x' + 10 - m) * 10 ^ k
  | otherwise = (x' - m) * 10 ^ k
  where
    x' = x `div` 10 ^ k
    m = x' `mod` 10

-- 浮動小数点を経由せず、小数点を四捨五入して整数にする割り算
roundDiv :: (Integral a) => a -> a -> a
roundDiv a b = (2 * a + b) `div` (2 * b)

sumFromTo :: (Integral a) => a -> a -> a
sumFromTo a b
  | b < a = 0
  | otherwise = (b - a + 1) * (a + b) `div` 2

sumFromToMod :: IntMod -> IntMod -> IntMod
sumFromToMod a b = (b - a + 1) * (a + b) * invIntMod 2

-- 初項a、公差 d、項数 n の等差数列の和
seqSum :: (Integral a) => a -> a -> a -> a
seqSum a d n = (n * (2 * a + (n - 1) * d)) `div` 2

-- 繰り返し二乗法による累乗 / 法を指定 (Python の pow 相当)
-- ex) powMod 2 60 998244353
powMod :: (Integral a1, Integral a2) => a1 -> a2 -> a1 -> a1
powMod _ 0 _ = 1
powMod x n m
  | even n = (t * t) `mod` m
  | otherwise = (((x * t) `mod` m) * t) `mod` m
  where
    t = powMod x (n `div` 2) m

-- >>> average 4 [3, 4, 7, 7]
-- (5,6)
average :: (Integral a, Foldable t) => a -> t a -> (a, a)
average n as = do
  let (p, q) = sum as `divMod` n
  (p, if q == 0 then p else p + 1)

-- >>> median [1, 3, 3, 4]
-- 3
median :: (Integral a) => [a] -> a
median xs
  | even n = (xs' !! (n `div` 2 - 1) + xs' !! (n `div` 2)) `div` 2
  | otherwise = xs' !! (n `div` 2)
  where
    n = length xs
    xs' = sort xs

divisors :: Int -> [Int]
divisors n = sort' . concat . mapMaybe f $ takeWhile (\a -> a * a <= n) [1 ..]
  where
    f a = do
      let (b, q) = n `divMod` a
      if q == 0
        then Just (bool [a, b] [a] (a == b))
        else Nothing

primeFactorize :: Int -> [Int]
primeFactorize n = unfoldr f (n, 2)
  where
    f (1, _) = Nothing
    f (m, p)
      | p ^ (2 :: Int) > m = Just (m, (1, m))
      | r == 0 = Just (p, (q, p))
      | otherwise = f (m, p + 1)
      where
        (q, r) = m `divMod` p

isSquareNumber :: (Integral a) => a -> Bool
isSquareNumber n = do
  let root = floor @Double (sqrt (fromIntegral n))
   in root * root == n

-- 平方数因子 (n に掛けたら平方数になる因数) を返す
-- 第一引数は素因数分解の関数
-- >>> squareFactor (primeFactorize) 12
-- 3
squareFactor :: (Int -> [Int]) -> Int -> Int
squareFactor factorizeF n =
  let bucket = accumIntMap (+) 0 [(p, 1 :: Int) | p <- factorizeF n]
   in (product . IM.keys . IM.filter odd) bucket

eratosthenes :: Int -> UArray Int Bool
eratosthenes n = runSTUArray $ do
  ps <- newArray (0, n) True
  mapM_ (\i -> writeArray ps i False) [0, 1]

  forM_ [2 .. n] $ \p -> do
    isPrime <- readArray ps p
    when isPrime $ do
      mapM_ (\i -> writeArray ps i False) [(p * 2), (p * 3) .. n]

  return ps

primes :: Int -> [Int]
primes n = [p | (p, isPrime) <- assocs (eratosthenes n), isPrime]

-- osa_k法 (高速素因数分解)
minFactorSieve :: Int -> UArray Int Int
minFactorSieve n = runSTUArray $ do
  minFactor <- newListArray (0, n) [0 .. n]

  forM_ [2 .. n] $ \i -> do
    !mf <- readArray minFactor i
    when (i == mf) $ do
      forM_ [(i * 2), (i * 3) .. n] $ \j -> do
        writeArray minFactor j i

  return minFactor

-- osa_k法による高速素因数分解
factorize :: UArray Int Int -> Int -> [Int]
factorize mf = unfoldr f
  where
    f k
      | k < 2 = Nothing
      | otherwise = do
          let p = mf ! k
          Just (p, k `div` p)

-- 度数法 -> 弧度法
toRadians :: (Floating a) => a -> a
toRadians d = d * (pi / 180)

-- ベクトルを反時計回りに alpha 回転 / 回転量は弧度法で指定
rotateCCWVec :: (RealFloat a) => (a, a) -> a -> (a, a)
rotateCCWVec (x, y) alpha = (r * cos (theta + alpha), r * sin (theta + alpha))
  where
    theta = atan2 y x
    r = sqrt $ x ^ (2 :: Int) + y ^ (2 :: Int)

-- 並行根
isqrt :: (Integral a) => a -> a
isqrt n
  | n < 0 = error "isqrt: negative input"
  | n < 2 = n
  | otherwise = go (n `div` 2)
  where
    go x =
      let x' = (x + n `div` x) `div` 2
       in if x' >= x then x else go x'

-- 立方根
-- 3乗してる都合上 n が大きいとオーバーフローする / Integer を使う方が無難
isqrt3 :: (Integral p) => p -> p
isqrt3 n = let (_, ok) = bisect (0, n + 1) (\x -> x * x * x > n) in ok

{-- digits --}

toBinary :: Int -> [Bool]
toBinary = unfoldr f
  where
    f 0 = Nothing
    f i = Just (q == 1, p)
      where
        (p, q) = i `divMod` 2

toDigits :: (Integral a) => a -> a -> [a]
toDigits _ 0 = [0]
toDigits n a = reverse $ unfoldr f a
  where
    f 0 = Nothing
    f x = Just (q, p)
      where
        (p, q) = divMod x n

fromDigits :: (Foldable t, Num a) => a -> t a -> a
fromDigits n = foldl' (\acc b -> acc * n + b) 0

{-- Data.Buffer --}

-- from https://github.com/cojna/iota/blob/master/src/Data/Buffer.hs
data Buffer s a = Buffer
  { bufferVars :: !(VUM.MVector s Int),
    internalBuffer :: !(VUM.MVector s a),
    internalBufferSize :: !Int
  }

_bufferFrontPos :: Int
_bufferFrontPos = 0

_bufferBackPos :: Int
_bufferBackPos = 1

-- type Queue s a = Buffer s a
newBufferAsQueue :: (VU.Unbox a, PrimMonad m) => Int -> m (Buffer (PrimState m) a)
newBufferAsQueue n = Buffer <$> VUM.replicate 2 0 <*> VUM.unsafeNew n <*> pure n

-- type Deque s a = Buffer s a
newBufferAsDeque :: (VU.Unbox a, PrimMonad m) => Int -> m (Buffer (PrimState m) a)
newBufferAsDeque n =
  Buffer
    <$> VUM.replicate 2 n
    <*> VUM.unsafeNew (2 * n)
    <*> pure (2 * n)

popFrontBuf :: (VU.Unbox a, PrimMonad m) => Buffer (PrimState m) a -> m (Maybe a)
popFrontBuf Buffer {bufferVars, internalBuffer} = do
  f <- VUM.unsafeRead bufferVars _bufferFrontPos
  b <- VUM.unsafeRead bufferVars _bufferBackPos
  if f < b
    then do
      VUM.unsafeWrite bufferVars _bufferFrontPos (f + 1)
      pure <$> VUM.unsafeRead internalBuffer f
    else return Nothing
{-# INLINE popFrontBuf #-}

pushFrontBuf :: (VU.Unbox a, PrimMonad m) => a -> Buffer (PrimState m) a -> m ()
pushFrontBuf x Buffer {bufferVars, internalBuffer} = do
  f <- VUM.unsafeRead bufferVars _bufferFrontPos
  VUM.unsafeWrite bufferVars _bufferFrontPos (f - 1)
  assert (f > 0) $ do
    VUM.unsafeWrite internalBuffer (f - 1) x
{-# INLINE pushFrontBuf #-}

pushBackBuf :: (VU.Unbox a, PrimMonad m) => a -> Buffer (PrimState m) a -> m ()
pushBackBuf x Buffer {bufferVars, internalBuffer, internalBufferSize} = do
  b <- VUM.unsafeRead bufferVars _bufferBackPos
  VUM.unsafeWrite bufferVars _bufferBackPos (b + 1)
  assert (b < internalBufferSize) $ do
    VUM.unsafeWrite internalBuffer b x
{-# INLINE pushBackBuf #-}

{-- graph --}

graph :: (Ix v) => (v, v) -> [(v, e)] -> Array v [e]
graph = accumArray (flip (:)) []

invGraph :: (Ix i) => (i, i) -> [(i, i)] -> Array i [i]
invGraph b uvs = accumArray (flip (:)) [] b $ map swap uvs

graph2 :: (Ix i) => (i, i) -> [(i, i)] -> Array i [i]
graph2 b uvs = accumArray (flip (:)) [] b $ concatMap (\uv -> [uv, swap uv]) uvs

wGraph :: (Ix v) => (v, v) -> [((v, v), e)] -> Array v [(v, e)]
wGraph b uvs = accumArray (flip (:)) [] b xs
  where
    xs = map (\((u, v), w) -> (u, (v, w))) uvs

invWGraph :: (Ix v) => (v, v) -> [((v, v), e)] -> Array v [(v, e)]
invWGraph b uvs = accumArray (flip (:)) [] b xs
  where
    xs = map (\((u, v), w) -> (v, (u, w))) uvs

wGraph2 :: (Ix v) => (v, v) -> [((v, v), e)] -> Array v [(v, e)]
wGraph2 b uvs = accumArray (flip (:)) [] b xs
  where
    xs = concatMap (\((u, v), w) -> [(u, (v, w)), (v, (u, w))]) uvs

imGraph :: (Foldable t) => t (IS.Key, a) -> IM.IntMap [a]
imGraph = accumIntMap (flip (:)) []

imGraph2 :: (Foldable t) => t (IS.Key, IS.Key) -> IM.IntMap [IS.Key]
imGraph2 uvs = accumIntMap (flip (:)) [] $ concatMap (\uv -> [uv, swap uv]) uvs

{-- グラフ探索 --}

topSortM :: (MArray IOUArray Bool m, Ix v) => (v -> [v]) -> (v, v) -> [v] -> m [v]
topSortM nextStates b vs = do
  let s = (IS.fromList . map ix) vs
  visited <- newArray @IOUArray b False
  vs' <- foldM (f visited) [] vs
  -- DFS で vs に含まれない頂点も辿るのでそれは取り除く
  return $ filter (\v -> IS.member (ix v) s) vs'
  where
    ix = index b

    f visited order v = do
      seen <- readArray visited v
      if seen
        then return order
        else dfsM nextStates visited order v

    dfsM :: (MArray a Bool m, Ix v) => (v -> [v]) -> a v Bool -> [v] -> v -> m [v]
    dfsM nextStates_ visited order v = do
      writeArray visited v True
      order' <- foldM step order (nextStates_ v)
      return (v : order') -- 帰りがけ順にオーダーを記録
      where
        step context u = do
          seen <- readArray visited u
          if seen
            then return context
            else dfsM nextStates_ visited context u

-- ex) sccM g
-- [[10,8,9,11],[5,6,7,4],[2,3,1]]
sccM :: (Ix v, IArray a [v]) => a v [v] -> IO [[v]]
sccM g = do
  let g' = reverseGraph g

  (_, ros) <- componentsM (g !) (bounds g) $ range (bounds g)
  (cs, _) <- componentsM (g' !) (bounds g) $ concat ros

  return cs
  where
    reverseGraph g_ = accumArray @Array (flip (:)) [] (bounds g_) $ do
      (v, us) <- assocs g_
      [(u, v) | u <- us]

    -- 訪問済み頂点を共有しながら全ての連結成分を訪問
    -- 帰りがけ順オーダーも記録
    componentsM :: (Ix v) => (v -> [v]) -> (v, v) -> [v] -> IO ([[v]], [[v]])
    componentsM nextStates (l, u_) vs = do
      visited <- newArray @IOUArray (l, u_) False
      foldM (f visited) ([], []) vs
      where
        f visited context@(cs, ros) v = do
          flag <- readArray visited v
          if flag
            then return context
            else do
              (path, ro) <- dfsM nextStates visited ([], []) v
              return (path : cs, (v : ro) : ros) -- 最後にスタート地点を加える
    dfsM :: (MArray a Bool m, Ix v) => (v -> [v]) -> a v Bool -> ([v], [v]) -> v -> m ([v], [v])
    dfsM nextStates visited (path, ro) v = do
      writeArray visited v True
      (path', ro') <- foldM f (v : path, ro) (nextStates v)
      return (path', v : ro')
      where
        f context u = do
          seen <- readArray visited u
          if seen
            then return context
            else dfsM nextStates visited context u

dfs :: (Ix v) => (v -> [v]) -> (v, v) -> ([v], IS.IntSet) -> v -> ([v], IS.IntSet)
dfs nextStates b (path, visited) v =
  let context = (v : path, IS.insert (ix v) visited)
   in foldl' f context (nextStates v)
  where
    ix = index b

    f context u
      | IS.member (ix u) (snd context) = context
      | otherwise = dfs nextStates b context u

components :: (Ix v, Foldable t) => (v -> [v]) -> (v, v) -> t v -> [[v]]
components nextStates b vs = do
  let (cs, _) = foldl' f ([], IS.empty) vs
  cs
  where
    ix = index b

    f context@(cs, visited) v
      | IS.member (ix v) visited = context
      | otherwise = do
          let (path, visited') = dfs nextStates b ([], visited) v
          (path : cs, visited')

bfs :: (Ix v) => (v -> [v]) -> Int -> (v, v) -> [(v, Int)] -> UArray v Int
bfs nextStates initial b v0s = runSTUArray $ do
  dist <- newArray b initial

  for_ v0s $ \(v0, d0) -> do
    writeArray dist v0 d0

  aux (Seq.fromList [v0 | (v0, _) <- v0s]) dist
  return dist
  where
    aux Empty _ = return ()
    aux (v :<| queue) dist = do
      d <- readArray dist v
      us <- filterM (fmap (== initial) . readArray dist) (nextStates v)

      queue' <- foldForM queue us $ \q u -> do
        writeArray dist u (d + 1)
        return $ q |> u

      aux queue' dist

-- キューの実装に Data.Buffer を利用
bfsWithBuffer :: (Ix v, VUM.Unbox v) => (v -> [v]) -> Int -> (v, v) -> [(v, Int)] -> UArray v Int
bfsWithBuffer nextStates initial b v0s = runSTUArray $ do
  dist <- newArray b initial
  queue <- newBufferAsQueue (rangeSize b)

  for_ v0s $ \(v0, d0) -> do
    writeArray dist v0 d0
    pushBackBuf v0 queue

  aux queue dist
  return dist
  where
    aux queue dist = do
      !entry <- popFrontBuf queue

      case entry of
        Nothing -> return ()
        Just v -> do
          d <- readArray dist v
          us <- filterM (fmap (== initial) . readArray dist) (nextStates v)

          for_ us $ \ !u -> do
            writeArray dist u $! d + 1
            pushBackBuf u queue

          aux queue dist

-- Hashable なら何でも頂点にできる BFS
bfsWithHashMap :: (Hashable.Hashable v, Num d) => (v -> [v]) -> [(v, d)] -> HM.HashMap v d
bfsWithHashMap nextStates v0s = do
  let dist = HM.fromList v0s
  aux (Seq.fromList [v0 | (v0, _) <- v0s]) dist
  where
    aux Empty dist = dist
    aux (v :<| queue) dist = do
      let d = dist HM.! v
          us = [u | u <- nextStates v, not (HM.member u dist)]

      let (queue', dist') = foldFor' (queue, dist) us $ \(q, dst) u -> do
            (q |> u, HM.insert u (d + 1) dst)

      aux queue' dist'

bfs2 ::
  forall a v.
  (IArray a v, Ix v) =>
  (v -> [v]) -> -- 次の状態への遷移関数
  Int -> -- 距離の初期値
  v -> -- 親頂点の初期値
  (v, v) -> -- 状態空間の境界
  [(v, Int)] -> -- 状態遷移の開始頂点
  (UArray v Int, a v v) -- (経路数, 親頂点)
bfs2 nextStates initial root_ b v0s = runST $ do
  dist <- newArray b initial :: ST s (STUArray s v Int)
  parent <- newArray b root_ :: ST s (STArray s v v)

  for_ v0s $ \(v0, d0) -> do
    writeArray dist v0 d0

  aux (Seq.fromList [v0 | (v0, _) <- v0s]) (dist, parent)

  dist' <- freeze dist
  parent' <- freeze parent

  return (dist', parent')
  where
    aux Empty _ = return ()
    aux (v :<| queue) (dist, parent) = do
      d <- readArray dist v
      us <- filterM (fmap (== initial) . readArray dist) (nextStates v)

      queue' <- foldForM queue us $ \q u -> do
        writeArray dist u (d + 1)
        writeArray parent u v
        return $ q |> u

      aux queue' (dist, parent)

data OptimizationDijkstra = Minimize | Maximize deriving (Show, Eq)

dijkstra ::
  forall v w e.
  (Ix v, Show v, Num e, Ord e, Show e) =>
  (v -> [(v, w)]) -> -- 状態遷移関数。重み付きで次の頂点を返す
  (e -> w -> e) -> -- 結合関数。現コストに次の遷移コストを結合する際の計算。典型的には (+)
  OptimizationDijkstra -> -- 最小化 (Minimize) / 最大化 (Maximize)
  e -> -- コストのデフォルト。最小化 maxBound / 最大化 minBound
  (v, v) -> -- 状態空間の境界
  Int -> -- 辺のサイズ
  [(v, e)] -> -- 頂点の初期コスト
  Array v e
dijkstra nextStates reflect opt def b_ edgeSize v0s = runSTArray $ do
  dist <- newArray b_ def

  -- 初期状態でヒープが満タンになるケースでは領域が足りなくなるため、2倍確保
  -- 領域が2倍になっても O(n) 処理はないので問題ない
  heap <- newSTBH (2 * max (rangeSize b_) edgeSize) :: ST s (BinaryHeap (STArray s) (p, v) (STRef s))

  for_ v0s $ \(v0, d0) -> do
    modifyArray dist v0 (`combineF` d0)
    insertBH heap (adj d0, v0)

  aux dist heap >> return dist
  where
    adj = case opt of
      Minimize -> id
      Maximize -> negate
    {-# INLINE adj #-}

    cmp = case opt of
      Minimize -> (<)
      Maximize -> (>)
    {-# INLINE cmp #-}

    combineF = case opt of
      Minimize -> min
      Maximize -> max
    {-# INLINE combineF #-}

    aux dist heap = do
      entry <- popBH heap

      case entry of
        Nothing -> return ()
        Just (dv_, v) -> do
          let dv = adj dv_

          -- dist v `cmp` dv means (dv, v) is garbage
          garbage <- (`cmp` dv) <$> readArray dist v

          unless garbage $ do
            for_ (nextStates v) $ \(u, w) -> do
              du <- readArray dist u

              let dv' = reflect dv w

              when (dv' `cmp` du) $ do
                writeArray dist u $! dv'
                insertBH heap (adj dv', u)

          aux dist heap

{-- Ref --}

class (Monad m) => Ref r m where
  newRef :: a -> m (r a)
  readRef :: r a -> m a
  writeRef :: r a -> a -> m ()
  modifyRef' :: r a -> (a -> a) -> m ()

instance Ref IORef IO where
  newRef = newIORef
  readRef = readIORef
  writeRef = writeIORef
  modifyRef' = modifyIORef'

instance Ref (STRef s) (ST s) where
  newRef = newSTRef
  readRef = readSTRef
  writeRef = writeSTRef
  modifyRef' = modifySTRef'

{-- MArray based Binary Heap --}

data BinaryHeap a e r = BinaryHeap
  { entriesBH :: a Int e,
    sizeRefBH :: r Int
  }

newBH :: (MArray a e m, Ref r m) => Int -> m (BinaryHeap a e r)
newBH !n = BinaryHeap <$> newArray_ (0, max 0 (n - 1)) <*> newRef 0

-- FIXME: newBH のまま使いたいが...
newIOBH :: (MArray a e IO) => Int -> IO (BinaryHeap a e IORef)
newIOBH = newBH

newSTBH :: (MArray a e (ST s)) => Int -> ST s (BinaryHeap a e (STRef s))
newSTBH = newBH

insertBH :: (MArray a e m, Ref r m, Ord e) => BinaryHeap a e r -> e -> m ()
insertBH heap@BinaryHeap {..} x = do
  n <- readRef sizeRefBH

  writeArray entriesBH n $! x
  modifyRef' sizeRefBH (+ 1)

  _upheap heap n
{-# INLINE insertBH #-}

popBH :: (MArray a e m, Ref r m, Ord e) => BinaryHeap a e r -> m (Maybe e)
popBH heap@BinaryHeap {..} = do
  n <- readRef sizeRefBH
  if n > 0
    then do
      x <- readArray entriesBH 0
      swapArray entriesBH 0 (n - 1)
      modifyRef' sizeRefBH (+ (-1))
      _downheap heap 0
      return $ Just x
    else return Nothing
{-# INLINE popBH #-}

peekBH :: (MArray a e m, Ref r m, Ord e) => BinaryHeap a e r -> m (Maybe e)
peekBH BinaryHeap {..} = do
  n <- readRef sizeRefBH
  if n > 0
    then Just <$> readArray entriesBH 0
    else return Nothing
{-# INLINE peekBH #-}

_upheap :: (MArray a e m, Ord e) => BinaryHeap a e r -> Int -> m ()
_upheap BinaryHeap {..} k = do
  x <- readArray entriesBH k
  flip fix k $ \loop !i ->
    if i > 0
      then do
        let parent = (i - 1) `shiftR` 1
        p <- readArray entriesBH parent
        case compare p x of
          GT -> (writeArray entriesBH i $! p) >> loop parent
          _ -> writeArray entriesBH i $! x
      else writeArray entriesBH 0 $! x
{-# INLINE _upheap #-}

_downheap :: (MArray a e m, Ref r m, Ord e) => BinaryHeap a e r -> Int -> m ()
_downheap BinaryHeap {..} k = do
  x <- readArray entriesBH k
  n <- readRef sizeRefBH

  flip fix k $ \loop !i -> do
    let l = (i `shiftL` 1) .|. 1
        r = l + 1

    if n <= l
      then writeArray entriesBH i $! x
      else do
        xl <- readArray entriesBH l

        if r < n
          then do
            xr <- readArray entriesBH r

            case compare xr xl of
              LT -> case compare x xr of
                GT -> (writeArray entriesBH i $! xr) >> loop r
                _ -> writeArray entriesBH i $! x
              _ -> case compare x xl of
                GT -> (writeArray entriesBH i $! xl) >> loop l
                _ -> writeArray entriesBH i $! x
          else case compare x xl of
            GT -> (writeArray entriesBH i $! xl) >> loop l
            _ -> writeArray entriesBH i $! x
{-# INLINE _downheap #-}

{-- IArray --}

genArray :: (IArray a e, Ix i) => (i, i) -> (i -> e) -> a i e
genArray (l, u) f = listArray (l, u) $ map f $ range (l, u)
{-# INLINE genArray #-}

swapIArray :: (IArray a e, Ix i) => i -> i -> a i e -> a i e
swapIArray i j as = as // [(i, as ! j), (j, as ! i)]
{-# INLINE swapIArray #-}

-- 右上隅を基点としたグリッドの回転 (グリッドは正方形を前提とする)
rotateRightArray :: (IArray a e, Ix i, Num i) => a (i, i) e -> a (i, i) e
rotateRightArray grid = ixmap (bounds grid) (\(i, j) -> (u + l - j, i)) grid
  where
    ((l, _), (u, _)) = bounds grid

toBucket :: (Ix i) => (i, i) -> [i] -> UArray i Int
toBucket b xs = accumArray @UArray (+) (0 :: Int) b $ map (,1) xs

isSubArrayOfBy :: (Ix i, IArray a1 t1, IArray a2 t2) => (t1 -> t2 -> Bool) -> a1 i t1 -> a2 i t2 -> Bool
isSubArrayOfBy predicate a b = and [predicate e (b ! i) | (i, e) <- assocs a]

(!?) :: (IArray a e, Ix i) => a i e -> i -> Maybe e
(!?) arr i =
  let b = bounds arr
   in if inRange b i
        then Just (arr ! i)
        else Nothing
{-# INLINE (!?) #-}

findArrayIndex :: (IArray a e, Ix i) => (e -> Bool) -> a i e -> Maybe i
findArrayIndex f as = fst <$> findOnSnd f (assocs as)
{-# INLINE findArrayIndex #-}

findArrayIndices :: (IArray a e, Ix i) => (e -> Bool) -> a i e -> [i]
findArrayIndices predicate as = [i | (i, e) <- assocs as, predicate e]
{-# INLINE findArrayIndices #-}

countArrayBy :: (IArray a e, Ix i) => (e -> Bool) -> a i e -> Int
countArrayBy predicate as = sum [1 :: Int | (_, e) <- assocs as, predicate e]
{-# INLINE countArrayBy #-}

{-- MArray --}

modifyArray :: (MArray a e m, Ix i) => a i e -> i -> (e -> e) -> m ()
modifyArray ary ix f = do
  v <- readArray ary ix
  writeArray ary ix $! f v
{-# INLINE modifyArray #-}

swapArray :: (MArray a e m, Ix i) => a i e -> i -> i -> m ()
swapArray as i j = do
  a <- readArray as i
  b <- readArray as j
  writeArray as j $! a
  writeArray as i $! b
{-# INLINE swapArray #-}

updateArray :: (MArray a e m, Ix i) => (e -> e' -> e) -> a i e -> i -> e' -> m ()
updateArray f arr ix x = do
  v <- readArray arr ix
  writeArray arr ix $! f v x
{-# INLINE updateArray #-}

{-- bisect --}

-- | 左が false / 右が true で境界を引く
bisect :: (Integral a) => (a, a) -> (a -> Bool) -> (a, a)
bisect (ng, ok) f
  | abs (ok - ng) == 1 = (ng, ok)
  | f m = bisect (ng, m) f
  | otherwise = bisect (m, ok) f
  where
    m = (ok + ng) `div` 2

-- | 左が true / 右が false で境界を引く
bisect2 :: (Integral a) => (a, a) -> (a -> Bool) -> (a, a)
bisect2 (ok, ng) f
  | abs (ng - ok) == 1 = (ok, ng)
  | f m = bisect2 (m, ng) f
  | otherwise = bisect2 (ok, m) f
  where
    m = (ok + ng) `div` 2

bisectM :: (Monad m, Integral a) => (a, a) -> (a -> m Bool) -> m (a, a)
bisectM (ng, ok) f
  | abs (ok - ng) == 1 = return (ng, ok)
  | otherwise = do
      x <- f mid
      if x
        then bisectM (ng, mid) f
        else bisectM (mid, ok) f
  where
    mid = (ok + ng) `div` 2

lookupGE :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> Maybe e
lookupGE x xs = do
  let (_, ub) = bounds xs
      ok = boundGE x xs

  if ok == succ ub
    then Nothing
    else Just (xs ! ok)

lookupGT :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> Maybe e
lookupGT x xs = do
  let (_, ub) = bounds xs
      i = boundGT x xs

  if i == succ ub
    then Nothing
    else Just (xs ! i)

lookupLT :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> Maybe e
lookupLT x xs = do
  let (lb, _) = bounds xs
      i = boundLT x xs

  if i == pred lb
    then Nothing
    else Just (xs ! i)

lookupLE :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> Maybe e
lookupLE x xs = do
  let (lb, _) = bounds xs
      i = boundLE x xs

  if i == pred lb
    then Nothing
    else Just (xs ! i)

boundGE :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> i
boundGE x xs = do
  let (lb, ub) = bounds xs
      (_, !ok) = bisect (pred lb, succ ub) (\i -> xs ! i >= x)
  ok

boundGT :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> i
boundGT x xs = do
  let (lb, ub) = bounds xs
      (_, !ok) = bisect (pred lb, succ ub) (\i -> xs ! i > x)
  ok

boundLT :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> i
boundLT x xs = do
  let (lb, ub) = bounds xs
      (!ng, _) = bisect (pred lb, succ ub) (\i -> xs ! i >= x)
  ng

boundLE :: (IArray a e, Ix i, Integral i, Ord e) => e -> a i e -> i
boundLE x xs = do
  let (lb, ub) = bounds xs
      (!ng, _) = bisect (pred lb, succ ub) (\i -> xs ! i > x)
  ng

countInRange :: (IArray a e, Ix i, Integral i, Ord e) => (e, e) -> a i e -> i
countInRange (l, r) as = boundLE r as - boundLT l as

{-- DP --}

-- 畳み込みDP
-- 時間的遷移時に、状態空間が前の状態を引き継ぐ (accum)
-- ex) accumDP @UArray f max minBound (0, wx) [(0, 0)] wvs
accumDP ::
  ( IArray a e,
    Ix v,
    Eq e,
    Show e,
    Show v,
    Show (a v e),
    Foldable t
  ) =>
  ((v, e) -> x -> [(v, e')]) -> -- 状態遷移関数 f v / x をみて v の次の遷移可能性を返す
  (e -> e' -> e) -> -- 緩和の二項演算
  e -> -- 初期値 (0, minBound, maxBound, False など)
  (v, v) -> -- 状態空間の下界、上界
  [(v, e')] -> -- 開始時点の状態
  t x -> -- 入力 (時間遷移)
  a v e -- Array or UArray
accumDP f combine initial (l, u) v0s xs =
  let dp = accumArray combine initial (l, u) v0s
   in foldl' transition dp xs
  where
    transition dp x =
      accum combine dp $
        concatMap (filter (inRange (bounds dp) . fst) . (`f` x)) (assocs dp)
      where
        !_ = dbg ("dp", [(v, acc) | (v, acc) <- assocs dp, acc /= initial])

-- 畳み込みDP
-- 時間的遷移時に、状態空間は前の状態を引き継がず都度リセットされる (accumArray)
-- ex) accumArrayDP @UArray f max minBound (0, wx) [(0, 0)] wvs
accumArrayDP ::
  ( IArray a e,
    Ix v,
    Eq e,
    Show e,
    Show v,
    Show (a v e),
    Foldable t
  ) =>
  ((v, e) -> x -> [(v, e')]) -> -- 状態遷移関数 f v / x をみて v の次の遷移可能性を返す
  (e -> e' -> e) -> -- 緩和の二項演算
  e -> -- 初期値 (0, minBound, maxBound, False など)
  (v, v) -> -- 状態空間の下界、上界
  [(v, e')] -> -- 開始時点の状態
  t x -> -- 入力 (時間遷移)
  a v e -- Array or UArray
accumArrayDP f combine initial (l, u) v0s xs =
  let dp = accumArray combine initial (l, u) v0s
   in foldl' transition dp xs
  where
    transition dp x =
      accumArray combine initial (l, u) $
        concatMap (filter (inRange (bounds dp) . fst) . (`f` x)) (assocs dp)
      where
        !_ = dbg ("dp", [(v, acc) | (v, acc) <- assocs dp, acc /= initial])

-- 前方への単純な配るDP
-- 引数の最後は　DP の探索範囲。N - 1 で止めたいときなどもあるのに対応
-- ex) linearDP @UArray f (||) False (0, n) [(0, True)] [0 .. n - 1]
linearDP ::
  forall a v e.
  (IArray a e, Ix v) =>
  (v -> e -> [(v, e)]) ->
  (e -> e -> e) ->
  e ->
  (v, v) ->
  [(v, e)] ->
  [v] ->
  a v e
linearDP f op initial b v0s vs = runST $ do
  dp <- newArray b initial :: ST s (STArray s v e)

  for_ v0s $ \(v, x) -> do
    writeArray dp v $! x

  for_ vs $ \v -> do
    x <- readArray dp v
    for_ (f v x) $ \(u, x') -> do
      when (inRange b u) $ do
        updateArray op dp u x'

  freeze dp

-- STUArray ベースの linearDP
-- 値は Int に限定 (Unbox な値に拡げたいがうまくいってない)
linearDP' ::
  (Ix i, Foldable t) =>
  (i -> Int -> t (i, Int)) ->
  (Int -> Int -> Int) ->
  Int ->
  (i, i) ->
  t (i, Int) ->
  t i ->
  UArray i Int
linearDP' f op initial b v0s vs = runSTUArray $ do
  dp <- newArray b initial

  for_ v0s $ \(v, x) -> do
    writeArray dp v $! x

  for_ vs $ \v -> do
    x <- readArray dp v
    for_ (f v x) $ \(u, x') -> do
      when (inRange b u) $ do
        modifyArray dp u (`op` x')

  return dp

{-- しゃくとり法 --}

-- ex) shakutori @UArray continue op invOp 0 s
shakutori ::
  forall a e acc.
  (IArray a e) =>
  (acc -> e -> Bool) -> -- (acc -> e -> Bool) -> -- 区間を維持できる条件
  (acc -> e -> acc) -> -- アキュムレータと R 位置の値の二項演算 (R が右に1つ進むときの演算)
  (acc -> e -> acc) -> -- アキュムレータと L 位置の値の逆演算 (L が右に1つ進むときの演算)
  acc -> -- 単位元 (L == R になってアキュムレータがこの値にリセットされる)
  [e] -> -- 対象の系列
  [((Int, Int), acc)] -- [((i, i), acc)] -- 条件を満たす (L, R) 区間 + アキュムレータのリスト
shakutori continue op invOp identity xs = loop (1, 1) identity
  where
    n = length xs
    xa = listArray (1, n) xs :: a Int e
    getValue i = xa ! i

    loop (l, r) acc
      | r > n = []
      | l > n = []
      | continue acc (getValue r) = let acc' = op acc (getValue r) in ((l, r), acc') : loop (l, succ r) acc'
      | l == r = loop (succ l, succ r) identity
      | otherwise = loop (succ l, r) (invOp acc (xa ! l))

{-- IO UnionFind --}

data UnionFind a v = UnionFind
  { parentUF :: a v v, -- 親頂点 / -1 は代表元
    sizeUF :: IOUArray v Int, -- 集合サイズ (代表元で検索する)
    edgesUF :: IOUArray v Int, -- 辺のサイズ
    numComponentsUF :: IORef Int, -- 連結成分数
    repUF :: v -- 代表元 (representative element)
  }

newUF :: (MArray a v IO, Ix v) => (v, v) -> v -> IO (UnionFind a v)
newUF (l, u) rep =
  UnionFind
    <$> newArray (l, u) rep
    <*> newArray (l, u) 1
    <*> newArray (l, u) 0
    <*> newIORef (bool 0 (ix u + 1 - ix l) (u >= l))
    <*> pure rep
  where
    ix = index (l, u)

newListUF :: (MArray a v IO, Ix v, Foldable t) => (v, v) -> v -> t (v, v) -> IO (UnionFind a v)
newListUF b root_ uvs = do
  uf <- newUF b root_
  for_ uvs $ \(u, v) -> do
    unite uf u v
  return uf

rootUF :: (MArray a v m, Ix v) => UnionFind a v -> v -> m v
rootUF uf@UnionFind {parentUF, repUF} x = do
  !p <- readArray parentUF x
  if p == repUF
    then return x
    else do
      p' <- rootUF uf p
      writeArray parentUF x p'
      return p'

unite :: (MArray a v IO, Ix v) => UnionFind a v -> v -> v -> IO ()
unite uf@UnionFind {..} x y = do
  x' <- rootUF uf x
  y' <- rootUF uf y

  when (x' == y') $ do
    edgesX <- readArray edgesUF x'
    writeArray edgesUF x' (edgesX + 1)

  when (x' /= y') $ do
    sizeX <- readArray sizeUF x'
    sizeY <- readArray sizeUF y'
    edgesX <- readArray edgesUF x'
    edgesY <- readArray edgesUF y'

    -- 併合する毎に集合が一つ減る
    modifyIORef' numComponentsUF (+ (-1))

    if sizeX > sizeY
      then do
        writeArray parentUF y' x'
        writeArray sizeUF x' (sizeX + sizeY)
        writeArray edgesUF x' (edgesX + edgesY + 1)
      else do
        writeArray parentUF x' y'
        writeArray sizeUF y' (sizeX + sizeY)
        writeArray edgesUF y' (edgesX + edgesY + 1)

isSame :: (MArray a v m, Ix v) => UnionFind a v -> v -> v -> m Bool
isSame uf x y = (==) <$> rootUF uf x <*> rootUF uf y

getSizeUF :: (MArray a v IO, Ix v) => UnionFind a v -> v -> IO Int
getSizeUF uf@UnionFind {sizeUF} x = do
  y <- rootUF uf x
  readArray sizeUF y

getEdgeSizeUF :: (MArray a v IO, Ix v) => UnionFind a v -> v -> IO Int
getEdgeSizeUF uf@UnionFind {edgesUF} x = do
  y <- rootUF uf x
  readArray edgesUF y

getComponentsUF :: (MArray a v m, Ix v) => UnionFind a v -> m [[v]]
getComponentsUF uf@UnionFind {parentUF} = do
  bx <- getBounds parentUF
  vs <-
    mapM
      ( \v -> do
          r <- rootUF uf v
          return (r, v)
      )
      (range bx)

  let cs = filter (not . null) . elems $ accumArray @Array (flip (:)) [] bx vs
  return cs

getParentsUF :: (IArray ia v, MArray a v m, Ix v) => UnionFind a v -> m (ia v v)
getParentsUF UnionFind {parentUF} = freeze parentUF

getSizesUF :: (MArray a v IO, Ix v) => UnionFind a v -> IO (UArray v Int)
getSizesUF UnionFind {sizeUF} = freeze sizeUF

getEdgeSizesUF :: (MArray a v IO, Ix v) => UnionFind a v -> IO (UArray v Int)
getEdgeSizesUF UnionFind {edgesUF} = freeze edgesUF

getNumComponentsUF :: (MArray a v IO, Ix v) => UnionFind a v -> IO Int
getNumComponentsUF UnionFind {numComponentsUF} = readIORef numComponentsUF

{-- Set --}

deleteViewSet :: (Ord p) => (p -> Set.Set p -> Maybe p) -> p -> Set.Set p -> ([p], Set.Set p)
deleteViewSet f x s0 = loop s0 []
  where
    loop s acc = case f x s of
      Just v ->
        let s' = Set.delete v s
         in loop s' (v : acc)
      Nothing -> (acc, s)

deleteGEViewSet, deleteGTViewSet, deleteLEViewSet, deleteLTViewSet :: (Ord p) => p -> Set.Set p -> ([p], Set.Set p)
deleteGEViewSet = deleteViewSet Set.lookupGE
deleteGTViewSet = deleteViewSet Set.lookupGT
deleteLEViewSet = deleteViewSet Set.lookupLE
deleteLTViewSet = deleteViewSet Set.lookupLT

{-- IntSet --}

deleteInRangeViewIS :: (IS.Key, IS.Key) -> IS.IntSet -> ([IS.Key], IS.IntSet)
deleteInRangeViewIS (l, r) = loop []
  where
    loop acc s = case IS.lookupGE l s of
      Just v | inRange (l, r) v -> loop (v : acc) (IS.delete v s)
      _ -> (acc, s)

deleteViewIS :: (Int -> IS.IntSet -> Maybe IS.Key) -> Int -> IS.IntSet -> ([IS.Key], IS.IntSet)
deleteViewIS f x s0 = loop s0 []
  where
    loop s acc = case f x s of
      Just v ->
        let s' = IS.delete v s
         in loop s' (v : acc)
      Nothing -> (acc, s)

deleteGEViewIS, deleteGTViewIS, deleteLEViewIS, deleteLTViewIS :: Int -> IS.IntSet -> ([IS.Key], IS.IntSet)
deleteGEViewIS = deleteViewIS IS.lookupGE
deleteGTViewIS = deleteViewIS IS.lookupGT
deleteLEViewIS = deleteViewIS IS.lookupLE
deleteLTViewIS = deleteViewIS IS.lookupLT

{-- IntMultiSet --}

data IntMultiSet = IntMultiSet
  { sizeMS :: !Int, -- 集合に含まれる値の個数 O(1)
    distinctSizeMS :: !Int, -- 集合に含まれる値の種類数 O(1)
    mapMS :: IM.IntMap Int
  }
  deriving (Eq)

instance Show IntMultiSet where
  show = show . toListMS

fromListMS :: [Int] -> IntMultiSet
fromListMS = foldl' (flip insertMS) emptyMS

toListMS :: IntMultiSet -> [Int]
toListMS IntMultiSet {mapMS = m} = concatMap @[] (\(k, v) -> replicate v k) (IM.toList m)

emptyMS :: IntMultiSet
emptyMS = IntMultiSet 0 0 IM.empty

singletonMS :: Int -> IntMultiSet
singletonMS x = insertMS x emptyMS

insertMS :: Int -> IntMultiSet -> IntMultiSet
insertMS x (IntMultiSet size dSize m)
  | IM.member x m = IntMultiSet (size + 1) dSize m'
  | otherwise = IntMultiSet (size + 1) (dSize + 1) m'
  where
    m' = IM.insertWith (+) x 1 m

-- x を n 個追加する
-- >>> insertNMS 5 2 (fromListMS [0])
-- [0,2,2,2,2,2]
insertNMS :: Int -> Int -> IntMultiSet -> IntMultiSet
insertNMS n x (IntMultiSet size dSize m)
  | IM.member x m = IntMultiSet (size + n) dSize m'
  | otherwise = IntMultiSet (size + n) (dSize + 1) m'
  where
    m' = IM.insertWith @Int (+) x n m

deleteMS :: Int -> IntMultiSet -> IntMultiSet
deleteMS x s@(IntMultiSet size dSize m)
  | IM.member x m =
      let m' = IM.update @Int (\k -> let k' = k - 1 in bool Nothing (Just k') $ k' > 0) x m
       in if IM.member x m'
            then IntMultiSet (size - 1) dSize m'
            else IntMultiSet (size - 1) (dSize - 1) m'
  | otherwise = s

-- x を n 個削除する
-- 存在する個数以上を指定した場合は削除できる分だけ削除
-- delete 2 100 で 100 を 2 個削除
-- >>> deleteNMS 1 2 (fromListMS [0, 2, 2])
-- [0,2]
deleteNMS :: Int -> Int -> IntMultiSet -> IntMultiSet
deleteNMS n x s@(IntMultiSet size dSize m)
  | IM.member x m = do
      let remain = m IM.! x
          m' = IM.update @Int (\k -> let k' = k - n in bool Nothing (Just k') $ k' > 0) x m
       in if IM.member x m'
            then IntMultiSet (size - n) dSize m'
            else IntMultiSet (size - remain) (dSize - 1) m'
  | otherwise = s

deleteMinMS :: IntMultiSet -> IntMultiSet
deleteMinMS s = deleteMS (findMinMS s) s

deleteMaxMS :: IntMultiSet -> IntMultiSet
deleteMaxMS s = deleteMS (findMaxMS s) s

deleteFindMinMS :: IntMultiSet -> (Int, IntMultiSet)
deleteFindMinMS s = (findMinMS s, deleteMinMS s)

deleteFindMaxMS :: IntMultiSet -> (Int, IntMultiSet)
deleteFindMaxMS s = (findMaxMS s, deleteMaxMS s)

unionWithMS :: (Int -> Int -> Int) -> IntMultiSet -> IntMultiSet -> IntMultiSet
unionWithMS f (IntMultiSet _ _ a) (IntMultiSet _ _ b) = IntMultiSet (IM.foldl' (+) 0 ab) (IM.size ab) ab
  where
    ab = IM.unionWith f a b

unionMS :: IntMultiSet -> IntMultiSet -> IntMultiSet
unionMS = unionWithMS (+)
{-# INLINE unionMS #-}

differenceWithMS :: (Int -> Int -> Maybe Int) -> IntMultiSet -> IntMultiSet -> IntMultiSet
differenceWithMS f (IntMultiSet _ _ a) (IntMultiSet _ _ b) = IntMultiSet (IM.foldl' (+) 0 c) (IM.size c) c
  where
    c = IM.differenceWith f a b

findMinMS :: IntMultiSet -> Int
findMinMS IntMultiSet {mapMS = m} = (fst . IM.findMin) m

findMaxMS :: IntMultiSet -> Int
findMaxMS IntMultiSet {mapMS = m} = (fst . IM.findMax) m

memberMS :: Int -> IntMultiSet -> Bool
memberMS x IntMultiSet {mapMS = m} = IM.member x m

notMemberMS :: Int -> IntMultiSet -> Bool
notMemberMS x IntMultiSet {mapMS = m} = IM.notMember x m

countMS :: Int -> IntMultiSet -> Int
countMS x IntMultiSet {mapMS = m} = IM.findWithDefault 0 x m

lookupLTMS :: Int -> IntMultiSet -> Maybe Int
lookupLTMS x IntMultiSet {mapMS = m} = (fmap fst . IM.lookupLT x) m

lookupGTMS :: Int -> IntMultiSet -> Maybe Int
lookupGTMS x IntMultiSet {mapMS = m} = (fmap fst . IM.lookupGT x) m

lookupLEMS :: Int -> IntMultiSet -> Maybe Int
lookupLEMS x IntMultiSet {mapMS = m} = (fmap fst . IM.lookupLE x) m

lookupGEMS :: Int -> IntMultiSet -> Maybe Int
lookupGEMS x IntMultiSet {mapMS = m} = (fmap fst . IM.lookupGE x) m

elemsMS :: IntMultiSet -> [Int]
elemsMS IntMultiSet {mapMS = m} = IM.elems m

nullMS :: IntMultiSet -> Bool
nullMS IntMultiSet {mapMS = m} = IM.null m

-- x 以下の値を k 個取得する。k 個ない場合は Nothing
-- >>> topKLE 3 3 (fromListMS [0, 0, 2, 2, 5])
-- Just [2,2,0]
topKLE :: Int -> Int -> IntMultiSet -> Maybe [Int]
topKLE k x set = aux [] x
  where
    aux xs x_ = do
      v <- lookupLEMS x_ set

      let i = countMS v set
          l = length xs

          -- ここで i 全部展開する必要はない
          ys = xs ++ replicate (min (k - l) i) v

      if length ys < k
        then aux ys (v - 1)
        else return $ take k ys

-- x 以上の値を k 個取得する。k 個ない場合は Nothing
-- >>> topKGE 3 1 (fromListMS [0, 0, 2, 2, 5])
-- Just [2,2,5]
topKGE :: Int -> Int -> IntMultiSet -> Maybe [Int]
topKGE k x set = aux [] x
  where
    aux xs x_ = do
      v <- lookupGEMS x_ set

      let i = countMS v set
          l = length xs
          ys = xs ++ replicate (min (k - l) i) v

      if length ys < k
        then aux ys (v + 1)
        else return $ take k ys

-- x 以上の値で k 番目の値を取得する
-- >>> findKthGE 1 4 (fromListMS [0, 0, 2, 2, 5])
-- Just 5
findKthGE :: Int -> Int -> IntMultiSet -> Maybe Int
findKthGE x k set = aux 0 x
  where
    aux cnt x_ = do
      v <- lookupGEMS x_ set

      let i = countMS v set
          cnt' = cnt + min (k - cnt) i

      if cnt' < k
        then aux cnt' (v + 1)
        else return v

deleteViewMS :: (Int -> IntMultiSet -> Maybe Int) -> Int -> IntMultiSet -> ([(Int, Int)], IntMultiSet)
deleteViewMS f x s0 = loop s0 []
  where
    loop s acc = case f x s of
      Just v ->
        let cnt = countMS v s
            s' = deleteNMS cnt v s
         in loop s' ((v, cnt) : acc)
      Nothing -> (acc, s)

deleteGEViewMS, deleteGTViewMS, deleteLEViewMS, deleteLTViewMS :: Int -> IntMultiSet -> ([(Int, Int)], IntMultiSet)
deleteGEViewMS = deleteViewMS lookupGEMS
deleteGTViewMS = deleteViewMS lookupGTMS
deleteLEViewMS = deleteViewMS lookupLEMS
deleteLTViewMS = deleteViewMS lookupLTMS

{-- Binary Indexed Tree --}

type BIT a v = a Int v

newBIT :: (MArray a v m, Num v, Monad m) => (Int, Int) -> m (BIT a v)
newBIT (l, u) = newArray (l, u + 1) 0

newListBIT :: (MArray a v m, Num v) => (Int, Int) -> [v] -> m (BIT a v)
newListBIT (l, u) as = do
  tree <- newBIT (l, u)
  for_ (indexed l as) $ \(i, a) -> do
    incrementBIT tree i a
  return tree

incrementBIT :: (MArray a v m, Num v, Monad m) => BIT a v -> Int -> v -> m ()
incrementBIT tree i v = do
  (_, n) <- getBounds tree

  flip fix (n, i + 1) $ \loop (k, !j) -> do
    when (j <= n) $ do
      updateArray (+) tree j v
      loop (k, j + (j .&. (-j)))

-- 0 から i までの区間和
readBIT :: (MArray a v m, Num v, Monad m) => BIT a v -> Int -> m v
readBIT tree i =
  flip fix (0, i) $ \loop (!acc, !j) -> do
    if j < 1
      then return acc
      else do
        x <- readArray tree j
        loop (acc + x, j - (j .&. (-j)))

-- l から r までの区間和 (半開区間で指定する [1, 6) が欲しいなら rangeSumBIT bit 1 6)
rangeSumBIT :: (MArray a v m, Num v, Monad m) => BIT a v -> Int -> Int -> m v
rangeSumBIT tree l r = do
  sl <- readBIT tree l
  sr <- readBIT tree r
  return $ sr - sl

{-- 転倒数 --}

-- >>> countInversion [7,6,5,4,3,2,1]
-- 21
-- >>> countInversion [20,19,2,1]
-- 6
countInversion :: (VUM.Unbox a, Ord a) => [a] -> Int
countInversion xs = runST $ do
  -- xs 内に被りがあっても大丈夫にする
  let xs' =
        VU.map fst $
          VU.modify (VAI.sortBy (\(x, y) (x', y') -> compare (y, x) (y', x'))) $
            VU.indexed $
              VU.fromList xs

      k = VU.length xs'

  tree <- newBIT (0, k - 1) :: ST s (BIT (STUArray s) Int)

  VU.foldM (f tree) 0 xs'
  where
    f tree !acc x = do
      v <- readBIT tree x
      incrementBIT tree x 1
      return $ acc + x - v

{-- TreeGraph --}

-- IArray と同様のインタフェースになっている

data TreeGraph a v e = TreeGraph
  { nextStatesTG :: v -> [v],
    boundsTG :: (v, v),
    distTG :: UArray v Int,
    parentTG :: UArray v v,
    valuesTG :: a v e
  }

_treeGraphWithValues :: (IArray a e, Ix v, Num v, IArray UArray v) => (v -> [v]) -> v -> a v e -> TreeGraph a v e
_treeGraphWithValues nextStates r values =
  TreeGraph nextStates (bounds values) dist parent values
  where
    (dist, parent) = bfs2 nextStates (-1) (-1) (bounds values) [(r, 0)]

instance (IArray a e, Ix v, Show v, Show e) => Show (TreeGraph a v e) where
  show = show . assocsTree

treeGraph ::
  (IArray a e, IArray UArray v, Ix v, Num v) =>
  (v -> [v]) ->
  v ->
  (v, v) ->
  e ->
  TreeGraph a v e
treeGraph nextStates r b initial =
  _treeGraphWithValues nextStates r $ accumArray const initial b []

listTree :: (IArray a e, IArray UArray v, Ix v, Num v) => (v -> [v]) -> v -> (v, v) -> [e] -> TreeGraph a v e
listTree nextStates r b =
  _treeGraphWithValues nextStates r . listArray b

genTree :: (IArray a e, IArray UArray v, Ix v, Num v) => (v -> [v]) -> v -> (v, v) -> (v -> e) -> TreeGraph a v e
genTree nextStates r b =
  _treeGraphWithValues nextStates r . genArray b

accumArrayTree :: (IArray a e, Ix v, Num v, IArray UArray v) => (v -> [v]) -> v -> (e -> e' -> e) -> e -> (v, v) -> [(v, e')] -> TreeGraph a v e
accumArrayTree nextStates r combine initial b =
  _treeGraphWithValues nextStates r . accumArray combine initial b

assocsTree :: (IArray a e, Ix v) => TreeGraph a v e -> [(v, e)]
assocsTree TreeGraph {valuesTG} = assocs valuesTG

accumTree :: (IArray a e, Ix v) => (e -> e' -> e) -> TreeGraph a v e -> [(v, e')] -> TreeGraph a v e
accumTree combine tree@TreeGraph {valuesTG} ies = tree {valuesTG = accum combine valuesTG ies}

foldTreeTopDown ::
  forall a i e.
  (IArray a e, IArray UArray i, Ix i) =>
  (e -> e -> e) ->
  TreeGraph a i e ->
  a i e
foldTreeTopDown f TreeGraph {..} = runST $ do
  let vs = (map fst . sortOn snd) (assocs distTG)

  dp <- thaw valuesTG :: ST s (STArray s i e)

  for_ vs $ \v -> do
    let us = filter (/= parentTG ! v) $ nextStatesTG v
    acc <- readArray dp v
    for_ us $ \u -> do
      x <- readArray dp u
      writeArray dp u (f acc x)

  freeze dp

-- 親から子へ一度に配るDP
-- f の引数は f acc us で、us は頂点番号 (他と IF が違うのに注意)
-- 戻り値は accumArray 的に [(u, val), ..] で返す
accumTreeTopDown ::
  forall a i e t b.
  (Foldable t, IArray a e, IArray UArray i, Ix i) =>
  (e -> [i] -> t (i, b)) ->
  (e -> b -> e) ->
  TreeGraph a i e ->
  a i e
accumTreeTopDown f combine TreeGraph {..} = runST $ do
  let vs = (map fst . sortOn snd) (assocs distTG)

  dp <- thaw valuesTG :: ST s (STArray s i e)

  for_ vs $ \v -> do
    let us = filter (/= parentTG ! v) (nextStatesTG v)
    acc <- readArray dp v
    for_ (f acc us) $ \(u, x') -> do
      modifyArray dp u (`combine` x')

  freeze dp

foldTreeBottomUp ::
  forall a i e.
  (IArray a e, IArray UArray i, Ix i) =>
  (e -> e -> e) ->
  TreeGraph a i e ->
  a i e
foldTreeBottomUp f TreeGraph {..} = runST $ do
  let vs = (map fst . sortOn (Down . snd)) (assocs distTG)

  dp <- thaw valuesTG :: ST s (STArray s i e)

  for_ (init vs) $ \v -> do
    acc <- readArray dp (parentTG ! v)
    x <- readArray dp v
    writeArray dp (parentTG ! v) (f x acc)

  freeze dp

-- 子から親に一度に集める DP (もらう DP)
accumTreeBottomUp ::
  forall a i e.
  (IArray a e, IArray UArray i, Ix i) =>
  ([e] -> e -> e) ->
  TreeGraph a i e ->
  a i e
accumTreeBottomUp f TreeGraph {..} = runST $ do
  let vs = (map fst . sortOn (Down . snd)) (assocs distTG)

  dp <- thaw valuesTG :: ST s (STArray s i e)

  for_ vs $ \v -> do
    let us = filter (/= parentTG ! v) (nextStatesTG v)

    -- 子から親に集める (もらうDP)
    acc <- readArray dp v
    xs <- traverse (readArray dp) us
    writeArray dp v (f xs acc)

  freeze dp

{-- GridSet --}

fromArrayGS :: (IArray a e, Ix v) => (e -> Bool) -> a v e -> Set.Set v
fromArrayGS f grid = Set.fromList $ map fst $ filterOnSnd f $ assocs grid

normalizeGS :: (Ord a, Num a) => Set.Set (a, a) -> Set.Set (a, a)
normalizeGS s = do
  let (i, j) = Set.findMin s
  Set.map (\(i', j') -> (i', j') - (i - 1, j - 1)) s

rotateRightGS :: (Ord a, Num a) => Set.Set (a, a) -> Set.Set (a, a)
rotateRightGS = Set.map rotateRight

rotateLeftGS :: (Ord a, Num a) => Set.Set (a, a) -> Set.Set (a, a)
rotateLeftGS = Set.map rotateLeft

-- 右上隅を基点にした90度回転 (正方形を前提)
rotateTopRightGS :: (Ord a, Num a) => a -> Set.Set (a, a) -> Set.Set (a, a)
rotateTopRightGS n = Set.map (\(i, j) -> (n + 1 - j, i))

rotatesGS :: (Ord a, Num a) => Set.Set (a, a) -> [Set.Set (a, a)]
rotatesGS s = take 4 $ iterate' rotateRightGS s

isShiftedGS :: (Ord a, Num a) => Set.Set (a, a) -> Set.Set (a, a) -> Bool
isShiftedGS s1 s2
  | Set.size s1 /= Set.size s2 = False
  | Set.null s1 && Set.null s2 = True
  | otherwise = do
      let diff = zipWith (-) (Set.toList s1) (Set.toList s2)
      Set.size (Set.fromList diff) == 1

{-- 2D CumSum --}

-- グリッドの配列から二次元累積和を構築する (速度を稼ぐため MArray を利用)
-- ((1,1), (h, w)) -> ((1, 1), (h + 1, w + 1)) になる (+1 は scanl の 0 が追加される)
fromArrayCS :: UArray (Int, Int) Int -> UArray (Int, Int) Int
fromArrayCS as = runSTUArray $ do
  s <- newArray b (0 :: Int)

  for_ (range (bounds as)) $ \(i, j) -> do
    writeArray s (i + 1, j + 1) (as ! (i, j))

  for_ [(i, j) | i <- [lh .. uh + 1], j <- [lw .. uw]] $ \(i, j) -> do
    x <- readArray s (i, j)
    modifyArray s (i, j + 1) (+ x)

  for_ [(i, j) | j <- [lw .. uw + 1], i <- [lh .. uh]] $ \(i, j) -> do
    x <- readArray s (i, j)
    modifyArray s (i + 1, j) (+ x)

  return s
  where
    ((lh, lw), (uh, uw)) = bounds as
    b = ((lh, lw), (uh + 1, uw + 1))

-- 矩形部分領域計算の抽象
rectRangeQuery :: (Num a1) => ((a2, b) -> a1) -> (a2, b) -> (a2, b) -> a1
rectRangeQuery f (a, b) (c, d) = f (c, d) - f (a, d) - f (c, b) + f (a, b)

-- 左上 (a, b) 右下 (c, d) に対する2次元累積和をクエリする
-- 右半開区間でクエリする (例) queryCS s (a, b) (c + 1, d + 1)
rectRangeSum :: UArray (Int, Int) Int -> (Int, Int) -> (Int, Int) -> Int
-- FIXME: 本当はこう定義したいが関数呼び出しにより速度が犠牲になる
-- rectRangeSum s = rectRangeQuery (s !)
rectRangeSum s (a, b) (c, d) = s ! (c, d) - s ! (a, d) - s ! (c, b) + s ! (a, b)

{-- 3D CumSum --}

-- 三次元配列から三次元累積和を構築する (速度を稼ぐため MArray を利用)
-- ((1,1,1), (h, w, d)) -> ((1,1,1), (h+1, w+1, d+1)) になる (+1 は scanl の 0 が追加される)
fromArrayCuboidCS :: UArray (Int, Int, Int) Int -> UArray (Int, Int, Int) Int
fromArrayCuboidCS as = runSTUArray $ do
  s <- newArray b (0 :: Int)

  for_ (range (bounds as)) $ \(x, y, z) -> do
    writeArray s (x + 1, y + 1, z + 1) (as ! (x, y, z))

  for_ [(x, y, z) | x <- [lh .. uh + 1], y <- [lw .. uw], z <- [ld .. ud + 1]] $ \(x, y, z) -> do
    val <- readArray s (x, y, z)
    modifyArray s (x, y + 1, z) (+ val)

  for_ [(x, y, z) | x <- [lh .. uh], y <- [lw .. uw + 1], z <- [ld .. ud + 1]] $ \(x, y, z) -> do
    val <- readArray s (x, y, z)
    modifyArray s (x + 1, y, z) (+ val)

  for_ [(x, y, z) | x <- [lh .. uh + 1], y <- [lw .. uw + 1], z <- [ld .. ud]] $ \(x, y, z) -> do
    val <- readArray s (x, y, z)
    modifyArray s (x, y, z + 1) (+ val)

  return s
  where
    ((lh, lw, ld), (uh, uw, ud)) = bounds as
    b = ((lh, lw, ld), (uh + 1, uw + 1, ud + 1))

-- 左上奥 (a, b, c) 右下手前 (d, e, f) に対する3次元累積和をクエリする / 半開区間
-- ex) cuboidRangeSum cs (lx, ly, lz) (rx +1, ry + 1, rz + 1)
cuboidRangeSum :: UArray (Int, Int, Int) Int -> (Int, Int, Int) -> (Int, Int, Int) -> Int
cuboidRangeSum s (a, b, c) (d, e, f) = s ! (d, e, f) - s ! (a, e, f) - s ! (d, b, f) - s ! (d, e, c) + s ! (a, b, f) + s ! (a, e, c) + s ! (d, b, c) - s ! (a, b, c)

{-- Data.Sequence --}

pushFrontSeq :: a -> Seq a -> Seq a
pushFrontSeq x xs = x <| xs
{-# INLINE pushFrontSeq #-}

pushBackSeq :: Seq a -> a -> Seq a
pushBackSeq xs x = xs |> x
{-# INLINE pushBackSeq #-}

viewFrontSeq :: Seq a -> Maybe a
viewFrontSeq Empty = Nothing
viewFrontSeq (x :<| _) = Just x
{-# INLINE viewFrontSeq #-}

viewBackSeq :: Seq a -> Maybe a
viewBackSeq Empty = Nothing
viewBackSeq (_ :|> x) = Just x
{-# INLINE viewBackSeq #-}

popFrontSeq :: Seq a -> Maybe (a, Seq a)
popFrontSeq Empty = Nothing
popFrontSeq (x :<| xs) = Just (x, xs)
{-# INLINE popFrontSeq #-}

popBackSeq :: Seq b -> Maybe (Seq b, b)
popBackSeq Empty = Nothing
popBackSeq (xs :|> x) = Just (xs, x)
{-# INLINE popBackSeq #-}

headSeq :: Seq p -> p
headSeq xs = case viewFrontSeq xs of
  Just x -> x
  Nothing -> error "empty sequence"
{-# INLINE headSeq #-}

lastSeq :: Seq p -> p
lastSeq xs = case viewBackSeq xs of
  Just x -> x
  Nothing -> error "empty sequence"
{-# INLINE lastSeq #-}

{-- Segment Tree --}

data SegmentTree a e = SegmentTree
  { boundsST :: (Int, Int), -- (下界、上界)
    sizeST :: Int, -- ノードサイズ

    -- モノイド
    combineST :: e -> e -> e,
    identityST :: e,
    nodeST :: a Int e
  }

-- 要素数 n のセグメント木を作る
newST ::
  (MArray a e m) =>
  (e -> e -> e) -> -- 結合演算
  e -> -- 単位元
  (Int, Int) -> -- (上界, 下界)
  m (SegmentTree a e)
newST combine identity (lb, ub) = do
  let size = pow2 (rangeSize (lb, ub))
  node <- newArray (lb, lb + 2 * size) identity
  return $ SegmentTree (lb, ub) size combine identity node
  where
    pow2 :: Int -> Int
    pow2 x = until (>= x) (* 2) 1

-- リストで与えた初期値で木を初期化する
buildST :: (MArray a e m) => SegmentTree a e -> [e] -> m ()
buildST SegmentTree {boundsST = (lb, _), ..} xs = do
  for_ (zip [sizeST ..] xs) $ \(i, x) -> do
    writeArray nodeST (lb + i) x

  for_ [sizeST - 1, sizeST - 2 .. 0] $ \i -> do
    !nodeL <- readArray nodeST (lb + (i `shiftL` 1))
    !nodeR <- readArray nodeST (lb + (i `shiftL` 1 .|. 1))
    writeArray nodeST (lb + i) $! combineST nodeL nodeR

newListST :: (MArray a e m) => (e -> e -> e) -> e -> (Int, Int) -> [e] -> m (SegmentTree a e)
newListST combine identity b xs = do
  tree <- newST combine identity b
  buildST tree xs
  return tree

getElemsST :: (MArray a e m) => SegmentTree a e -> m [e]
getElemsST tree@SegmentTree {boundsST} = traverse (readST tree) (range boundsST)

getAssocsST :: (MArray a e m) => SegmentTree a e -> m [(Int, e)]
getAssocsST tree@SegmentTree {boundsST} =
  traverse
    ( \i -> do
        x <- rangeQueryST tree i (i + 1)
        return (i, x)
    )
    (range boundsST)

-- 一点更新
updateST :: (MArray a e m) => SegmentTree a e -> Int -> (e -> e) -> m ()
updateST SegmentTree {boundsST = (lb, _), ..} i f = do
  -- 葉の位置
  let i' = i + sizeST - lb

  -- 葉 (== 列の値) を更新
  x <- readArray nodeST (lb + i')
  writeArray nodeST (lb + i') $! f x

  -- 更新を親 (区間ノード) に伝搬させる
  flip fix i' $ \loop !cur -> do
    when (cur > 1) $ do
      let parent = cur `shiftR` 1 -- 親
      nodeL <- readArray nodeST (lb + (parent `shiftL` 1)) -- 左の子
      nodeR <- readArray nodeST (lb + (parent `shiftL` 1 .|. 1)) -- 右の子
      writeArray nodeST (lb + parent) $! combineST nodeL nodeR -- 子の値を結合して親が持つ
      loop parent
{-# INLINE updateST #-}

-- 区間取得
-- [l, r) は半開区間で指定する
rangeQueryST :: (MArray a b m) => SegmentTree a b -> Int -> Int -> m b
rangeQueryST SegmentTree {boundsST = (lb, _), ..} a b =
  fix
    ( \loop !l !r !accL !accR ->
        if l < r
          then do
            vl <- readArray nodeST (lb + l)
            vr <- readArray nodeST (lb + r - 1)

            let (!l', !r', !accL', !accR') = case ((l .&. 1) == 1, (r .&. 1) == 1) of
                  (True, True) -> (l + 1, r - 1, combineST accL vl, combineST vr accR)
                  (True, False) -> (l + 1, r, combineST accL vl, accR)
                  (False, True) -> (l, r - 1, accL, combineST vr accR)
                  (False, False) -> (l, r, accL, accR)

            loop (l' `shiftR` 1) (r' `shiftR` 1) accL' accR'
          else do
            return $ combineST accL accR
    )
    (a + sizeST - lb)
    (b + sizeST - lb)
    identityST
    identityST
{-# INLINE rangeQueryST #-}

readST :: (MArray a b m) => SegmentTree a b -> Int -> m b
readST seg i = rangeQueryST seg i (i + 1)
{-# INLINE readST #-}

{-- MVector Lazy Segment Tree --}

-- 0-based index
data LazySegmentTree m e e' = LazySegmentTree
  { nLST :: Int, -- 要素数
    sizeLST :: Int, -- ノードサイズ
    heightLST :: Int, -- 木の高さ

    -- 値のモノイド
    combineLST :: e -> e -> e,
    identityLST :: e,
    nodeLST :: VUM.MVector (PrimState m) e,
    -- 作用のモノイド
    combineLazy :: e' -> e' -> e', -- 合成 (遅延操作を貯める)
    identityLazy :: e',
    lazyLST :: VUM.MVector (PrimState m) e',
    reflectLST :: e -> e' -> e -- 写像 (遅延操作のデータへの反映)
  }

newLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  Int ->
  (e -> e -> e) ->
  e ->
  (e' -> e' -> e') ->
  e' ->
  (e -> e' -> e) ->
  m (LazySegmentTree m e e')
newLST n combine identity combineLazy identityLazy reflect = do
  let size = pow2 n
      height = log2GE size

  node <- VUM.replicate (2 * size) identity
  lazy <- VUM.replicate (2 * size) identityLazy
  return $
    LazySegmentTree
      n
      size
      height
      combine
      identity
      node
      combineLazy
      identityLazy
      lazy
      reflect
  where
    pow2 :: Int -> Int
    pow2 1 = 2
    pow2 x = until (>= x) (* 2) 1

buildLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  [e] ->
  m ()
buildLST LazySegmentTree {..} xs = do
  for_ (zip [sizeLST ..] xs) $ \(i, x) -> do
    VUM.write nodeLST i x

  for_ [sizeLST - 1, sizeLST - 2 .. 1] $ \i -> do
    dataL <- VUM.read nodeLST (i `shiftL` 1)
    dataR <- VUM.read nodeLST (i `shiftL` 1 .|. 1)
    VUM.write nodeLST i $! combineLST dataL dataR

newListLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  Int ->
  (e -> e -> e) ->
  e ->
  (e' -> e' -> e') ->
  e' ->
  (e -> e' -> e) ->
  [e] ->
  m (LazySegmentTree m e e')
newListLST n combine identity combineLazy identityLazy reflect xs = do
  tree <- newLST n combine identity combineLazy identityLazy reflect
  buildLST tree xs
  return tree

getElemsLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  m [e]
getElemsLST tree@LazySegmentTree {nLST} = traverse (readLST tree) [0 .. (nLST - 1)]

getAssocsLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  m [(Int, e)]
getAssocsLST tree@LazySegmentTree {nLST} =
  traverse
    ( \i -> do
        x <- readLST tree i
        return (i, x)
    )
    [0 .. (nLST - 1)]

_propagateLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  m ()
_propagateLST LazySegmentTree {..} i = do
  for_ [heightLST, heightLST - 1 .. 1] $ \h -> do
    let parent = (i + sizeLST) `shiftR` h
    x <- VUM.read lazyLST parent

    when (x /= identityLazy) $ do
      VUM.modify lazyLST (`combineLazy` x) (parent `shiftL` 1)
      VUM.modify lazyLST (`combineLazy` x) (parent `shiftL` 1 .|. 1)
      VUM.modify nodeLST (`reflectLST` x) parent
      VUM.write lazyLST parent identityLazy
{-# INLINE _propagateLST #-}

_updateFromBottomLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  m ()
_updateFromBottomLST LazySegmentTree {..} i = do
  flip fix (i + sizeLST) $ \loop !cur -> do
    when (cur > 1) $ do
      let parent = cur `shiftR` 1
      dataL <- VUM.read nodeLST (parent `shiftL` 1)
      dataR <- VUM.read nodeLST (parent `shiftL` 1 .|. 1)
      lazyL <- VUM.read lazyLST (parent `shiftL` 1)
      lazyR <- VUM.read lazyLST (parent `shiftL` 1 .|. 1)
      VUM.write nodeLST parent $!
        combineLST (reflectLST dataL lazyL) (reflectLST dataR lazyR)
      loop parent
{-# INLINE _updateFromBottomLST #-}

-- 区間更新 [a, b)
rangeUpdateLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  Int ->
  e' ->
  m ()
rangeUpdateLST tree@LazySegmentTree {..} a b x = do
  _propagateLST tree a
  _propagateLST tree (b - 1)

  fix
    ( \loop !l !r -> do
        when (l < r) $ do
          l' <-
            if (l .&. 1) == 1
              then do
                VUM.modify lazyLST (`combineLazy` x) l
                return (l + 1)
              else return l

          r' <-
            if (r .&. 1) == 1
              then do
                VUM.modify lazyLST (`combineLazy` x) (r - 1)
                return (r - 1)
              else return r

          loop (l' `shiftR` 1) (r' `shiftR` 1)
    )
    (a + sizeLST)
    (b + sizeLST)

  _updateFromBottomLST tree a
  _updateFromBottomLST tree (b - 1)
{-# INLINE rangeUpdateLST #-}

-- 区間取得 [a, b)
rangeQueryLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  Int ->
  m e
rangeQueryLST tree@LazySegmentTree {..} a b = do
  _propagateLST tree a
  _propagateLST tree (b - 1)

  fix
    ( \loop !l !r !accL !accR ->
        if l < r
          then do
            dataL <- VUM.read nodeLST l
            dataR <- VUM.read nodeLST (r - 1)
            lazyL <- VUM.read lazyLST l
            lazyR <- VUM.read lazyLST (r - 1)

            let valueL = reflectLST dataL lazyL
                valueR = reflectLST dataR lazyR

            let (!l', !r', !accL', !accR') = case ((l .&. 1) == 1, (r .&. 1) == 1) of
                  (True, True) -> (l + 1, r - 1, combineLST accL valueL, combineLST valueR accR)
                  (True, False) -> (l + 1, r, combineLST accL valueL, accR)
                  (False, True) -> (l, r - 1, accL, combineLST valueR accR)
                  (False, False) -> (l, r, accL, accR)

            loop (l' `shiftR` 1) (r' `shiftR` 1) accL' accR'
          else do
            return $ combineLST accL accR
    )
    (a + sizeLST)
    (b + sizeLST)
    identityLST
    identityLST
{-# INLINE rangeQueryLST #-}

-- 1点取得 / 1点更新
readLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  m e
readLST tree i = rangeQueryLST tree i (i + 1)
{-# INLINE readLST #-}

updateLST ::
  (PrimMonad m, VUM.Unbox e, VUM.Unbox e', Eq e, Eq e') =>
  LazySegmentTree m e e' ->
  Int ->
  e' ->
  m ()
updateLST tree i = rangeUpdateLST tree i (i + 1)
{-# INLINE updateLST #-}

{-- Data.Bits --}

msb :: (Integral b, Num a, Ord a) => a -> b
msb = log2LE

bitLength :: (Integral b, Num a, Ord a) => a -> b
bitLength = succ . msb

{-- BitSet --}

-- 1-based index
newtype BitSet = BitSet Int deriving (Eq, Ord, Ix, Enum)

emptyBS :: BitSet
emptyBS = BitSet 0
{-# INLINE emptyBS #-}

-- 要素数 N のとき、全要素をもった集合を返す
fullBS :: Int -> BitSet
fullBS n = BitSet (2 ^ n - 1)
{-# INLINE fullBS #-}

unionBS :: BitSet -> BitSet -> BitSet
unionBS = coerce $ (.|.) @Int
{-# INLINE unionBS #-}

intersectionBS :: BitSet -> BitSet -> BitSet
intersectionBS = coerce $ (.&.) @Int
{-# INLINE intersectionBS #-}

insertBS :: Int -> BitSet -> BitSet
insertBS i (BitSet bits) = BitSet (setBit bits (i - 1))
{-# INLINE insertBS #-}

deleteBS :: Int -> BitSet -> BitSet
deleteBS i (BitSet bits) = BitSet (clearBit bits (i - 1))

memberBS :: Int -> BitSet -> Bool
memberBS i (BitSet bits) = testBit bits (i - 1)
{-# INLINE memberBS #-}

notMemberBS :: Int -> BitSet -> Bool
notMemberBS i = not . memberBS i
{-# INLINE notMemberBS #-}

singletonBS :: Int -> BitSet
singletonBS i = BitSet (setBit 0 (i - 1))
{-# INLINE singletonBS #-}

fromListBS :: [Int] -> BitSet
fromListBS = foldl' (flip insertBS) emptyBS
{-# INLINE fromListBS #-}

sizeBS :: BitSet -> Int
sizeBS = coerce $ popCount @Int
{-# INLINE sizeBS #-}

bitLengthBS :: BitSet -> Int
bitLengthBS (BitSet bits) = bitLength bits
{-# INLINE bitLengthBS #-}

-- >>> powersetBS (fromListBS [1,3])
-- [5, 4, 1]
-- >>> [ toListBS s | s <- powersetBS (fromListBS [1, 3])]
-- [[1,3], [3], [1]]
powersetBS :: BitSet -> [BitSet]
powersetBS (BitSet s) = emptyBS : unfoldr f s
  where
    f 0 = Nothing
    f i = Just (BitSet i, (i - 1) .&. s)

-- base を基準として s の冪集合を返す
-- powersetBS の補集合
-- >>> complementPowersetBS (fullBS 4) (fromListBS [1, 3])
-- [10,8,2]
-- >>> [ toListBS s | s <- complementPowersetBS (fullBS 4) (fromListBS [1, 3])]
-- [[2,4],[4],[2]]
complementPowersetBS :: BitSet -> BitSet -> [BitSet]
complementPowersetBS (BitSet base) (BitSet s) = powersetBS (BitSet (xor base s))

toListBS :: BitSet -> [Int]
toListBS (BitSet x) = reverse $ loop (1, x) []
  where
    loop (_, 0) xs = xs
    loop (i, v) xs
      | q == 0 = loop (i + 1, p) xs
      | otherwise = loop (i + 1, p) (i : xs)
      where
        (p, q) = v `divMod` 2

instance Show BitSet where
  showsPrec p xs =
    showParen (p > 10) $
      showString "fromList "
        . shows (toListBS xs)

-- TODO: findMin, findMax, lookup, difference etc

-- n をビットフラグと見立てた時の、ビットが立っている箇所の冪集合を10進で返す
-- >>> bitPowerset 7
-- >>> bitPowerset 5
-- >>> bitPowerset 3
-- [7,6,5,4,3,2,1]
-- [5,4,1]
-- [3,2,1]
bitPowerset :: Int -> [Int]
bitPowerset n = unfoldr f n
  where
    f 0 = Nothing
    f i = Just (i, (i - 1) .&. n)
{-# INLINE bitPowerset #-}

-- m を基準に整数 n のビットの立っていない箇所の冪集合を10進で返す
-- つまり bitPowerset の補集合
-- >>> bitComplementPowerset 7 1
-- >>> bitComplementPowerset 7 4
-- >>> bitComplementPowerset 7 5
-- [6,4,2]
-- [3,2,1]
-- [2]
bitComplementPowerset :: Int -> Int -> [Int]
bitComplementPowerset m n = bitPowerset (xor m n)

{-- 座標圧縮 --}

-- 座標圧縮 ··· (ランク -> 値) の UArray とインデックス関数を返す
-- ex) let (rank, ix) = zaatsuRank @UArray 1 as
--         i = ix x
zaatsuRank :: (IArray a e, VU.Unbox e, Ord e) => Int -> [e] -> (a Int e, e -> Int)
zaatsuRank i as = (rank, ix)
  where
    xs = indexed i (nubOrd' as)
    rank = array (i, i + length xs - 1) xs
    ix x = boundGE x rank

-- i はランキング一位の値。典型的には 0始まりか 1始まり
-- >>> zaatsu 0 [3,3,1,6,1]
-- [1,1,0,2,0]
zaatsu :: Int -> [Int] -> [Int]
zaatsu i as = [ix a | a <- as]
  where
    (_, ix) = zaatsuRank @UArray i as

{-- Utility --}

-- K ビットのビット列を固定幅キューとみなして後ろから値を push する
pushBackBit :: (Bits a, Num a) => Int -> a -> a -> a
pushBackBit k b bits = (shiftL bits 1 .|. b) .&. (bit k - 1)

-- N x N グリッドの対角線のマス
diagonals :: (Num b, Enum b) => b -> [[(b, b)]]
diagonals n = [zip [1 .. n] [1 .. n], zip [1 .. n] [n, n - 1 .. 1]]

-- >>> mex []
-- 0
-- >>> mex [1]
-- 0
-- >>> mex [0,1,2]
-- 3
-- >>> mex [0,1,50]
-- 2
mex :: [Int] -> Int
mex xs = fromJust $ find (`IS.notMember` set) [0 ..]
  where
    set = IS.fromList xs

{-- trace --}

dbg :: (Show a) => a -> ()
dbg = case getDebugEnv of
  Just _ -> (`traceShow` ())
  Nothing -> const ()

getDebugEnv :: Maybe String
getDebugEnv = unsafePerformIO (lookupEnv "DEBUG")
{-# NOINLINE getDebugEnv #-}

-- }}}
