use std::convert::TryFrom;
use std::path::Path;

use anyhow::Result;
use image::{self, imageops::FilterType};
use serde::Serialize;
use tch::{CModule, Kind, TchError, Tensor};

#[derive(Debug)]
pub struct MnistInput(Vec<f32>);

impl MnistInput {
    pub fn from_image_bytes(bytes: Vec<u8>) -> Result<Self> {
        const NORM_SCALE: f32 = 1. / 255.;
        let im = image::load_from_memory(&bytes)?
            .resize_exact(28, 28, FilterType::Nearest)
            .grayscale()
            .to_bytes()
            .into_iter()
            .map(|x| (x as f32) * NORM_SCALE)
            .collect::<Vec<f32>>();
        Ok(Self(im))
    }
}

impl TryFrom<MnistInput> for Tensor {
    type Error = TchError;

    fn try_from(value: MnistInput) -> Result<Self, Self::Error> {
        Tensor::f_of_slice(&value.0)?.f_reshape(&[1, 1, 28, 28])
    }
}

#[derive(Debug, Serialize)]
pub struct MnistPrediction {
    pub label: u8,
    pub confidence: f64,
}

pub struct MnistModel {
    model: CModule,
}

impl MnistModel {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let model = CModule::load(path)?;
        Ok(Self { model })
    }

    pub fn predict(&self, image: MnistInput) -> Result<MnistPrediction> {
        let tensor = Tensor::try_from(image)?;
        let output: Vec<f64> = self
            .model
            .forward_ts(&[tensor])?
            .softmax(-1, Kind::Double)
            .into();

        let mut confidence = 0f64;
        let mut label = 0u8;
        for (i, prob) in output.into_iter().enumerate() {
            if prob > confidence {
                confidence = prob;
                label = i as u8;
            }
        }

        Ok(MnistPrediction { label, confidence })
    }
}
