//  LIB.rs
//    by Lut99
//
//  Created:
//    13 Sep 2024, 10:57:52
//  Last edited:
//    13 Sep 2024, 16:30:11
//  Auto updated?
//    Yes
//
//  Description:
//!   Implements procedural macro for doing math on integer literals for
//!   use in Rust macros.
//

#![recursion_limit = "256"]

use std::fmt::{Display, Formatter, Result as FResult};
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::ToTokens;
use syn::spanned::Spanned as _;
use syn::token::Minus;
use syn::{BinOp, Error, Expr, ExprBinary, ExprCast, ExprLit, ExprTuple, ExprUnary, Lit, LitInt, Type, UnOp};


/***** HELPER MACROS *****/
/// Explodes two sets into a set of all possibilities.
macro_rules! cast_up_match {
    // API
    (
        ($self:ident, $target:ident) =>
        [$($variant:ident),+],
        [$(($kind:ident, $dtype:ty)),*]
    ) => {
        cast_up_match!(__expand
            ($self, $target) =>
            [$($variant),+],
            ([$(<$kind, $dtype>),*], [$(<$kind, $dtype>),*]),
            []
        )
    };

    // Base case: both lists are exhausted
    (__expand
        ($self:ident, $target:ident) =>
        [$variant:ident],
        ([], [$(<$fkind:ident, $fdtype:ty>),*]),
        [$(($rvariant:ident, $rkind:ident, $rdtype:ty)),*]
    ) => {
        match ($self, $target) {
            $((Self::$rvariant(var), IntegerKind::$rkind) => Self::$rkind(var as $rdtype),)*
        }
    };
    // First case: variant list has elements, but kind list is exhausted (reset it)
    (__expand
        ($self:ident, $target:ident) =>
        [$variant:ident $(, $tvariant:ident)+],
        ([], [$(<$fkind:ident, $fdtype:ty>),*]),
        [$(($rvariant:ident, $rkind:ident, $rdtype:ty)),*]
    ) => {
        cast_up_match!(__expand
            ($self, $target) =>
            [$($tvariant),*],
            ([$(<$fkind, $fdtype>),*], [$(<$fkind, $fdtype>),*]),
            [$(($rvariant, $rkind, $rdtype)),*]
        )
    };
    // Second case: both have elements
    (__expand
        ($self:ident, $target:ident) =>
        [$variant:ident $(, $tvariant:ident)*],
        ([<$kind:ident, $dtype:ty> $(, <$tkind:ident, $tdtype:ty>)*], [$(<$fkind:ident, $fdtype:ty>),*]),
        [$(($rvariant:ident, $rkind:ident, $rdtype:ty)),*]
    ) => {
        cast_up_match!(__expand
            ($self, $target) =>
            [$variant $(, $tvariant)*],
            ([$(<$tkind, $tdtype>),*], [$(<$fkind, $fdtype>),*]),
            [($variant, $kind, $dtype) $(, ($rvariant, $rkind, $rdtype))*]
        )
    };
}

/// Implementation for an arbitrary binary operator.
macro_rules! binop_impl {
    ($self:ident $op:tt $rhs:ident) => {
        // Figure out the common basis
        let target: Option<IntegerKind> = match ($self.kind(), $rhs.kind()) {
            (Some(lhsk), Some(rhsk)) => {
                if lhsk.size() > rhsk.size() {
                    if rhsk.is_signed() { Some(lhsk.into_signed()) } else { Some(lhsk) }
                } else {
                    if lhsk.is_signed() { Some(rhsk.into_signed()) } else { Some(rhsk) }
                }
            },
            (Some(lhsk), None) => Some(lhsk),
            (None, Some(rhsk)) => Some(rhsk),
            (None, None) => None,
        };

        // Upcast both to that
        let lhs: Self = $self.cast_to(target);
        let rhs: Self = $rhs.cast_to(target);

        // Do the math now
        match (lhs, rhs) {
            (Self::Auto(lhs), Self::Auto(rhs)) => Self::Auto(lhs $op rhs),

            (Self::U8(lhs), Self::U8(rhs)) => Self::U8(lhs $op rhs),
            (Self::U16(lhs), Self::U16(rhs)) => Self::U16(lhs $op rhs),
            (Self::U32(lhs), Self::U32(rhs)) => Self::U32(lhs $op rhs),
            (Self::U64(lhs), Self::U64(rhs)) => Self::U64(lhs $op rhs),
            (Self::U128(lhs), Self::U128(rhs)) => Self::U128(lhs $op rhs),
            (Self::USize(lhs), Self::USize(rhs)) => Self::USize(lhs $op rhs),

            (Self::I8(lhs), Self::I8(rhs)) => Self::I8(lhs $op rhs),
            (Self::I16(lhs), Self::I16(rhs)) => Self::I16(lhs $op rhs),
            (Self::I32(lhs), Self::I32(rhs)) => Self::I32(lhs $op rhs),
            (Self::I64(lhs), Self::I64(rhs)) => Self::I64(lhs $op rhs),
            (Self::I128(lhs), Self::I128(rhs)) => Self::I128(lhs $op rhs),
            (Self::ISize(lhs), Self::ISize(rhs)) => Self::ISize(lhs $op rhs),

            _ => unreachable!(),
        }
    };
}





/***** HELPERS *****/
/// Represents types of integers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum IntegerKind {
    U8,
    U16,
    U32,
    U64,
    U128,
    USize,

    I8,
    I16,
    I32,
    I64,
    I128,
    ISize,
}
impl IntegerKind {
    /// Returns the signed variant of this integer.
    #[inline]
    fn into_signed(self) -> Self {
        match self {
            Self::U8 => Self::I8,
            Self::U16 => Self::I16,
            Self::U32 => Self::I32,
            Self::U64 => Self::I64,
            Self::U128 => Self::I128,
            Self::USize => Self::ISize,

            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::ISize => self,
        }
    }



    /// Returns whether this one is signed or unsigned.
    #[inline]
    fn is_signed(&self) -> bool {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 | Self::U128 | Self::USize => false,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::I128 | Self::ISize => true,
        }
    }

    /// Returns the bitsize of this variant.
    #[inline]
    fn size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 => 16,
            Self::U32 | Self::I32 => 32,
            Self::U64 | Self::I64 => 64,
            Self::U128 | Self::I128 => 128,
            Self::USize => usize::BITS as usize,
            Self::ISize => isize::BITS as usize,
        }
    }
}



/// Represents a variable-sized integer.
#[derive(Debug)]
enum Integer {
    Auto(u64),

    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    USize(usize),

    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    ISize(isize),
}
impl Integer {
    /// Casts this integer to another type.
    ///
    /// # Arguments
    /// - `target`: The type to cast to. If [`None`], will not change the cast.
    #[inline]
    fn cast_to(self, target: Option<IntegerKind>) -> Self {
        match target {
            Some(target) => cast_up_match!(
                (self, target) =>
                [Auto, U8, U16, U32, U64, U128, USize, I8, I16, I32, I64, I128, ISize],
                [(U8, u8), (U16, u16), (U32, u32), (U64, u64), (U128, u128), (USize, usize), (I8, i8), (I16, i16), (I32, i32), (I64, i64), (I128, i128), (ISize, isize)]
            ),
            None => self,
        }
    }

    /// Returns the kind of integer, if known.
    #[inline]
    fn kind(&self) -> Option<IntegerKind> {
        match self {
            Self::Auto(_) => None,

            Self::U8(_) => Some(IntegerKind::U8),
            Self::U16(_) => Some(IntegerKind::U16),
            Self::U32(_) => Some(IntegerKind::U32),
            Self::U64(_) => Some(IntegerKind::U64),
            Self::U128(_) => Some(IntegerKind::U128),
            Self::USize(_) => Some(IntegerKind::USize),

            Self::I8(_) => Some(IntegerKind::I8),
            Self::I16(_) => Some(IntegerKind::I16),
            Self::I32(_) => Some(IntegerKind::I32),
            Self::I64(_) => Some(IntegerKind::I64),
            Self::I128(_) => Some(IntegerKind::I128),
            Self::ISize(_) => Some(IntegerKind::ISize),
        }
    }
}

// Formatting
impl Display for Integer {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FResult {
        match self {
            Self::Auto(val) => write!(f, "{val}"),

            Self::U8(val) => write!(f, "{val}u8"),
            Self::U16(val) => write!(f, "{val}u16"),
            Self::U32(val) => write!(f, "{val}u32"),
            Self::U64(val) => write!(f, "{val}u64"),
            Self::U128(val) => write!(f, "{val}u128"),
            Self::USize(val) => write!(f, "{val}usize"),

            Self::I8(val) => write!(f, "{val}i8"),
            Self::I16(val) => write!(f, "{val}i16"),
            Self::I32(val) => write!(f, "{val}i32"),
            Self::I64(val) => write!(f, "{val}i64"),
            Self::I128(val) => write!(f, "{val}i128"),
            Self::ISize(val) => write!(f, "{val}isize"),
        }
    }
}

// Arithmetic
impl Add for Integer {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        binop_impl! {self + rhs}
    }
}
impl Sub for Integer {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        binop_impl! {self - rhs}
    }
}
impl Mul for Integer {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        binop_impl! {self * rhs}
    }
}
impl Div for Integer {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        binop_impl! {self / rhs}
    }
}
impl Rem for Integer {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        binop_impl! {self % rhs}
    }
}

impl BitAnd for Integer {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        binop_impl! {self & rhs}
    }
}
impl BitOr for Integer {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        binop_impl! {self | rhs}
    }
}
impl BitXor for Integer {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: Self) -> Self::Output {
        binop_impl! {self ^ rhs}
    }
}
impl Shl for Integer {
    type Output = Self;

    #[inline]
    fn shl(self, rhs: Self) -> Self::Output {
        binop_impl! {self << rhs}
    }
}
impl Shr for Integer {
    type Output = Self;

    #[inline]
    fn shr(self, rhs: Self) -> Self::Output {
        binop_impl! {self >> rhs}
    }
}

// Conversion
impl TryFrom<LitInt> for Integer {
    type Error = Error;

    #[inline]
    fn try_from(value: LitInt) -> Result<Self, Self::Error> { Integer::try_from(&value) }
}
impl TryFrom<&LitInt> for Integer {
    type Error = Error;

    #[inline]
    fn try_from(value: &LitInt) -> Result<Self, Self::Error> {
        match value.suffix() {
            "" => Ok(Integer::Auto(value.base10_parse()?)),

            "u8" => Ok(Integer::U8(value.base10_parse()?)),
            "u16" => Ok(Integer::U16(value.base10_parse()?)),
            "u32" => Ok(Integer::U32(value.base10_parse()?)),
            "u64" => Ok(Integer::U64(value.base10_parse()?)),
            "u128" => Ok(Integer::U128(value.base10_parse()?)),
            "usize" => Ok(Integer::USize(value.base10_parse()?)),

            "i8" => Ok(Integer::I8(value.base10_parse()?)),
            "i16" => Ok(Integer::I16(value.base10_parse()?)),
            "i32" => Ok(Integer::I32(value.base10_parse()?)),
            "i64" => Ok(Integer::I64(value.base10_parse()?)),
            "i128" => Ok(Integer::I128(value.base10_parse()?)),
            "isize" => Ok(Integer::ISize(value.base10_parse()?)),

            other => Err(Error::new(value.span(), format!("Unknown literal suffix {other:?}"))),
        }
    }
}





/***** HELPER FUNCTIONS *****/
/// Evaluates a Rust expression.
///
/// This is also the entrypoint to the traversal tree.
///
/// # Arguments
/// - `expr`: The [`Expr`] to evaluate.
///
/// # Returns
/// A [`LitInt`] representing the evaluated value.
///
/// # Errors
/// This function errors if the input was not a valid integer expression.
#[inline]
fn evaluate_expr(expr: Expr) -> Result<Integer, Error> {
    match expr {
        Expr::Binary(bin) => evaluate_bin(bin),
        Expr::Cast(cast) => evaluate_cast(cast),
        Expr::Group(group) => evaluate_expr(*group.expr),
        Expr::Lit(lit) => evaluate_lit(lit),
        Expr::Paren(paren) => evaluate_expr(*paren.expr),
        Expr::Reference(rfr) => evaluate_expr(*rfr.expr),
        Expr::Tuple(tpl) => evaluate_tuple(tpl),
        Expr::Unary(una) => evaluate_una(una),

        // Anything else is not interesting
        expr => return Err(Error::new(expr.span(), "Cannot do arithmetic on non-integer expression")),
    }
}

/// Evaluates a Rust binary operation.
///
/// # Arguments
/// - `bin`: The [`ExprBinary`] to evaluate.
///
/// # Returns
/// An [`Integer`] representing the evaluated value.
///
/// # Errors
/// This function errors if the input was not a valid integer operation or the operands were not
/// valid integer expressions.
#[inline]
fn evaluate_bin(bin: ExprBinary) -> Result<Integer, Error> {
    match bin.op {
        BinOp::Add(_) => Ok(evaluate_expr(*bin.left)? + evaluate_expr(*bin.right)?),
        BinOp::Sub(_) => Ok(evaluate_expr(*bin.left)? - evaluate_expr(*bin.right)?),
        BinOp::Mul(_) => Ok(evaluate_expr(*bin.left)? * evaluate_expr(*bin.right)?),
        BinOp::Div(_) => Ok(evaluate_expr(*bin.left)? / evaluate_expr(*bin.right)?),
        BinOp::Rem(_) => Ok(evaluate_expr(*bin.left)? % evaluate_expr(*bin.right)?),

        BinOp::BitAnd(_) => Ok(evaluate_expr(*bin.left)? & evaluate_expr(*bin.right)?),
        BinOp::BitOr(_) => Ok(evaluate_expr(*bin.left)? | evaluate_expr(*bin.right)?),
        BinOp::BitXor(_) => Ok(evaluate_expr(*bin.left)? ^ evaluate_expr(*bin.right)?),
        BinOp::Shl(_) => Ok(evaluate_expr(*bin.left)? << evaluate_expr(*bin.right)?),
        BinOp::Shr(_) => Ok(evaluate_expr(*bin.left)? >> evaluate_expr(*bin.right)?),

        // Anything else is not interesting
        op => Err(Error::new(op.span(), "Unsupported binary operation")),
    }
}

/// Evaluates a Rust cast.
///
/// # Arguments
/// - `cast`: The [`ExprCast`] to evaluate.
///
/// # Returns
/// An [`Integer`] representing the evaluated value.
///
/// # Errors
/// This function errors if the casted expression was not a valid integer expression, or if the
/// outcome was not cast to an integer literal.
#[inline]
fn evaluate_cast(cast: ExprCast) -> Result<Integer, Error> {
    fn get_type_as_int(ty: Type) -> Result<IntegerKind, Error> {
        match ty {
            Type::Group(ty) => get_type_as_int(*ty.elem),
            Type::Ptr(ty) => get_type_as_int(*ty.elem),
            Type::Reference(ty) => get_type_as_int(*ty.elem),
            Type::Paren(ty) => get_type_as_int(*ty.elem),
            Type::Path(ty) => {
                if let Some(ident) = ty.path.get_ident() {
                    let ident: String = ident.to_string();
                    match ident.as_str() {
                        "u8" => Ok(IntegerKind::U8),
                        "u16" => Ok(IntegerKind::U16),
                        "u32" => Ok(IntegerKind::U32),
                        "u64" => Ok(IntegerKind::U64),
                        "u128" => Ok(IntegerKind::U128),
                        "usize" => Ok(IntegerKind::USize),

                        "i8" => Ok(IntegerKind::I8),
                        "i16" => Ok(IntegerKind::I16),
                        "i32" => Ok(IntegerKind::I32),
                        "i64" => Ok(IntegerKind::I64),
                        "i128" => Ok(IntegerKind::I128),
                        "isize" => Ok(IntegerKind::ISize),

                        other => Err(Error::new(ident.span(), format!("Unsupported type {other:?}"))),
                    }
                } else {
                    Err(Error::new(ty.span(), "Can only cast to types without a path of length 1 (e.g., `u8`"))
                }
            },
            Type::Tuple(mut ty) => {
                if ty.elems.len() == 1 {
                    get_type_as_int(ty.elems.pop().unwrap().into_value())
                } else {
                    Err(Error::new(ty.span(), "Can only cast to tuple of length 1"))
                }
            },

            // Anything else is not interesting
            ty => Err(Error::new(ty.span(), "Unsupported casting type")),
        }
    }

    // See if we can cast
    let target: IntegerKind = get_type_as_int(*cast.ty)?;
    let value: Integer = evaluate_expr(*cast.expr)?;
    Ok(value.cast_to(Some(target)))
}

/// Evaluates a Rust literal.
///
/// # Arguments
/// - `lit`: The [`ExprLit`] to evaluate.
///
/// # Returns
/// An [`Integer`] representing the evaluated value.
///
/// # Errors
/// This function errors if the input was not a valid integer literal.
#[inline]
fn evaluate_lit(lit: ExprLit) -> Result<Integer, Error> {
    match lit.lit {
        Lit::Int(int) => Integer::try_from(int),
        lit => return Err(Error::new(lit.span(), "Cannot do arithmetic on non-integer literal")),
    }
}

/// Evaluates a Rust tuple.
///
/// # Arguments
/// - `tuple`: The [`ExprTuple`] to evaluate.
///
/// # Returns
/// An [`Integer`] representing the evaluated value.
///
/// # Errors
/// This function errors if the input was a tuple with not exactly one expression, or that
/// expression was not a valid integer expression.
#[inline]
fn evaluate_tuple(mut tuple: ExprTuple) -> Result<Integer, Error> {
    if tuple.elems.len() == 1 {
        evaluate_expr(tuple.elems.pop().unwrap().into_value())
    } else {
        Err(Error::new(tuple.span(), "Tuples must have exactly one element"))
    }
}

/// Evaluates a Rust unary operator.
///
/// # Arguments
/// - `una`: The [`ExprUnary`] to evaluate.
///
/// # Returns
/// An [`Integer`] representing the evaluated value.
///
/// # Errors
/// This function errors if the input was not a valid integer operation or the operand was not a
/// valid integer expression.
#[inline]
fn evaluate_una(una: ExprUnary) -> Result<Integer, Error> {
    match una.op {
        UnOp::Deref(_) => evaluate_expr(*una.expr),
        UnOp::Neg(n) => evaluate_expr(Expr::Binary(ExprBinary {
            attrs: vec![],
            left:  Box::new(Expr::Lit(ExprLit { attrs: vec![], lit: Lit::Int(LitInt::new("0", n.span)) })),
            op:    BinOp::Sub(Minus { spans: [n.span] }),
            right: una.expr,
        })),
        op => Err(Error::new(op.span(), "Unsupported unary operation")),
    }
}





/***** LIBRARY *****/
/// Defines the procedural macro for doing Rust integer literal math.
///
/// Specifically, this macro supports a basic version of integer arithmetic on literals.
///
/// # Arguments
/// - `input`: The input stream containing an integer expression.
///
/// # Returns
/// A new [`TokenStream`] that encodes the result of the input literals.
///
/// # Examples
/// ```rust
/// use literal_arithmetic::calc;
///
/// // Tip: use the `cargo-expand` crate to verify these expands to the `2` literal
/// assert_eq!(calc!(1 + 1), 2);
/// assert_eq!(calc!(1 * 2), 2);
/// assert_eq!(calc!((2 + 2) / 2), 2);
/// assert_eq!(calc!((2 + 2usize) / 2), 2usize);
/// assert_eq!(calc!((2isize + 2usize) / 2), 2isize);
/// assert_eq!(calc!((2 + (2 as usize)) / 2), 2usize);
/// ```
#[inline]
#[proc_macro]
pub fn calc(input: TokenStream) -> TokenStream {
    let input: TokenStream2 = input.into();

    // Let's attempt to parse the input as an expression, first
    let expr: Expr = match syn::parse2(input) {
        Ok(expr) => expr,
        Err(err) => return err.into_compile_error().into(),
    };

    // Then attempt to evaluate the expression
    let span: Span = expr.span();
    let res: Integer = match evaluate_expr(expr) {
        Ok(res) => res,
        Err(err) => return err.into_compile_error().into(),
    };

    // OK, return the result
    let res: LitInt = LitInt::new(&res.to_string(), span);
    res.into_token_stream().into()
}
