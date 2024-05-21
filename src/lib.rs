use std::io::{Cursor, Read};

use anyhow::{bail, Context, Result};
use flate2::read::GzDecoder;
use tar::Archive;
use tracing::{span, trace, Level};

pub async fn load_model_files<T: ModelLoader>(
    mut loader: T,
    manifest: &T::ModelManifest,
) -> Result<T::LoadResult, anyhow::Error> {
    for file_name in T::ModelManifest::files_list() {
        let span = span!(Level::DEBUG, "load_file", file_name = file_name.as_str());
        let _guard = span.enter();
        let model_file = manifest.model_file(&file_name);
        match model_file {
            Ok(model_file) => {
                let bytes = load_file(&file_name).await?.bytes().await?;
                loader.load(model_file, bytes)?;
            }
            // because we read file from ModelFiles::files_list() we should never get here
            Err(_) => bail!("unknown file: {}", file_name),
        }
    }
    loader.finish().context("finishing model loading")
}

pub async fn load_model_bundle<T: ModelLoader>(
    path: &str,
    mut loader: T,
    manifest: &T::ModelManifest,
) -> Result<T::LoadResult, anyhow::Error> {
    let tar_gz = load_file(path).await?.bytes().await?;
    let tar = GzDecoder::new(Cursor::new(tar_gz));
    let mut archive = Archive::new(tar);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        trace!("found file in the bundle: {:?}", path);
        let path_name = path.to_string_lossy().to_string();
        let model_file = manifest.model_file(&path_name);
        match model_file {
            Ok(model_file) => {
                trace!("loading file: {:?}", path);
                let mut buf = Vec::new();
                entry.read_to_end(&mut buf).context("reading file")?;
                loader.load(model_file, buf)?;
            }
            Err(_) => {
                /* ignore unknown files */
                trace!("ignoring file: {:?}", path)
            }
        }
    }
    loader.finish().context("finishing model loading")
}

pub async fn load_model<R: ModelLoader>(
    loader: R,
    manifest: &R::ModelManifest,
) -> Result<R::LoadResult> {
    if manifest.bundle() {
        load_model_bundle(manifest.path(), loader, manifest)
            .await
            .map_err(Into::into)
    } else {
        load_model_files(loader, manifest).await.map_err(Into::into)
    }
}

async fn load_file(url: &str) -> Result<object_store::GetResult> {
    let (object_store, path) =
        object_store::parse_url(&url::Url::parse(url).context("parsing url")?)
            .context("object_store.parse")?;
    Ok(object_store.get(&path).await?)
}

pub trait ModelManifest {
    type ModelFile; // enum

    fn files_list() -> Vec<String>;
    fn model_file(&self, file_name: &str) -> Result<Self::ModelFile>;
    fn bundle(&self) -> bool;
    fn path(&self) -> &str;
}

pub trait ModelLoader {
    type LoadResult: Send + Sync + 'static;
    type ModelManifest: ModelManifest;
    type Error: std::error::Error + Send + Sync + 'static;

    fn load(
        &mut self,
        file_name: <Self::ModelManifest as ModelManifest>::ModelFile,
        reader: impl AsRef<[u8]> + Send + Sync + 'static,
    ) -> Result<(), Self::Error>;
    fn finish(self) -> Result<Self::LoadResult, Self::Error>;
}

/// usage example:
/// ```
/// model_manifest!(MyModelManifest, { Model => "model.pb", Labels => "labels.txt" });
/// ```
/// will generate:
/// ```
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// pub enum MyModelManifestFile {
///    Model,
///    Labels,
/// }
///
/// #[serde_inline_default::serde_inline_default]
/// #[derive(Clone, Debug, Default, serde::Deserialize)]
/// pub struct MyModelManifest {
///   pub model: String, // with serde default value "model.pb"
///   pub labels: String, // with serde default value "labels.txt"
/// }
/// ```
///
/// You will now have to implement `ModelLoader` for your model:
/// ```
/// impl ModelLoader for MyModel {
///    type LoadResult = MyModelResult;
///    type ModelManifest = MyModelManifest;
///    type Error = MyModelError;
///
///   fn load(&mut self, file_name: MyModelManifestFile, reader: impl MmapBytesReader) -> Result<(), Self::Error> {
///    match file_name {
///       MyModelManifestFile::Model => {
///           // load model from reader
///       }
///       MyModelManifestFile::Labels => {
///          // load labels from reader
///       }
///   }
///

#[macro_export]
macro_rules! model_manifest {
    ($name:ident, {$($file_name:ident => $default_file_name:literal),+ } ) => {
        paste::paste! {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub enum [< $name File>] {
                $($file_name),+
            }

            #[serde_inline_default]
            #[derive(Clone, Debug, serde::Deserialize)]
            pub struct $name {
                pub path: String,
                #[serde(default)]
                pub bundle: bool,
                $(
                    #[serde_inline_default($default_file_name.to_owned())]
                    pub [<$file_name:snake>]: String
                ),+
            }

            impl std::default::Default for $name {
                fn default() -> Self {
                    Self {
                        path: "".to_owned(),
                        bundle: false,
                        $([<$file_name:snake>]: $default_file_name.to_owned()),+
                    }
                }
            }

            impl $crate::ModelManifest for $name {
                type ModelFile = [< $name File> ];

                fn bundle(&self) -> bool {
                    self.bundle
                }

                fn path(&self) -> &str {
                    &self.path
                }

                fn files_list() -> Vec<String> {
                    vec![$($default_file_name.to_owned()),+]
                }

                fn model_file(&self, s: &str) -> Result<Self::ModelFile, anyhow::Error> {
                    tracing::trace!("checking file: {}, for self: {:?}", s, self);
                    $(
                        if self.[<$file_name:snake>] == s {
                            return Ok(Self::ModelFile::$file_name)
                        }
                    )+
                    return Err(anyhow::anyhow!("unknown file: {}", s))
                }
            }

        }
    };
}


pub mod prelude {
    pub use crate::ModelLoader;
    pub use crate::ModelManifest;
    pub use crate::model_manifest;

    pub use paste;
    pub use serde_inline_default::serde_inline_default;
    pub use serde;
}
