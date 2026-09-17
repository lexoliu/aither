//! Crate-internal macros.

/// Declare a string-valued wire enum whose unknown values round-trip
/// verbatim through an `Other(String)` variant.
///
/// ```ignore
/// string_enum! {
///     /// Doc comment.
///     pub enum GoalStatus {
///         /// Wire value `"active"`.
///         Active = "active",
///         Paused = "paused",
///     }
/// }
/// ```
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$vmeta:meta])*
                $variant:ident = $value:literal
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        $vis enum $name {
            $(
                $(#[$vmeta])*
                $variant,
            )*
            /// A value this crate does not model, preserved verbatim.
            Other(::std::string::String),
        }

        impl $name {
            /// The wire string for this value.
            #[must_use]
            pub const fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $value,)*
                    Self::Other(other) => other.as_str(),
                }
            }

            /// Build from a wire string; unknown values become `Other`.
            #[must_use]
            pub fn from_wire(value: ::std::string::String) -> Self {
                $(if value == $value {
                    return Self::$variant;
                })*
                Self::Other(value)
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                let value =
                    <::std::string::String as ::serde::Deserialize>::deserialize(deserializer)?;
                ::std::result::Result::Ok(Self::from_wire(value))
            }
        }

        impl ::std::convert::From<::std::string::String> for $name {
            fn from(value: ::std::string::String) -> Self {
                Self::from_wire(value)
            }
        }
    };
}

pub(crate) use string_enum;
