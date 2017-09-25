use std::fmt;
use std::error;

/// Generic error type of this library.
#[derive(Debug)]
pub struct Error {
    description: String,
}

impl Error {
    /// Create a new error message with the specified description.
    pub(crate) fn new<S>(s: S) -> Self
    where
        S: Into<String>,
    {
        Error { description: s.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.pad(&self.description)
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        &self.description
    }
}

#[cfg(test)]
mod test {
    use super::Error;
    use std::error::Error as ErrorTrait;

    #[test]
    fn new() {
        let e = Error::new("Test error");
        assert_eq!(e.description(), "Test error");
        assert!(e.cause().is_none());
    }
}
