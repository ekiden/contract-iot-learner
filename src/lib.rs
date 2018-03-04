#![feature(use_extern_macros)]

extern crate ndarray;
extern crate protobuf;
extern crate serde;
extern crate serde_cbor;
#[macro_use]
extern crate serde_derive;

extern crate rusty_machine;

extern crate ekiden_core_common;
extern crate ekiden_core_trusted;

extern crate iot_learner_api;

mod contract;
mod utils;
#[macro_use]
mod macros;

use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::prelude::*;

use ekiden_core_common::{Error, Result};
use ekiden_core_common::contract::{Address, Contract};
use ekiden_core_trusted::db::Db;
use ekiden_core_trusted::rpc::create_enclave_rpc;

use iot_learner_api::*;
use utils::{pack_proto, unpack_feature_matrix, unpack_feature_vector};

use contract::Learner;

// Create enclave RPC handlers.
with_api! {
    create_enclave_rpc!(api);
}

macro_rules! unpack {
    ($req:ident) => {
        {
            let state = Db::instance().get("state")?;
            let learner = Learner::<LinRegressor>::from_state(&state);
            if !Address::from($req.get_requester().to_string()).eq(learner.get_owner()?) {
                return Err(Error::new("Insufficient permissions."));
            }
            learner
        }
    }
}

fn create(req: &CreateRequest) -> Result<CreateResponse> {
    let learner = Learner::new(
        Address::from(req.get_requester().to_string()),
        LinRegressor::default(),
        req.get_inputs().to_vec(),
        req.get_targets().to_vec(),
    )?;
    Db::instance().set("state", learner.get_state())?;
    Ok(CreateResponse::new())
}

fn train(req: &TrainingRequest) -> Result<TrainingResponse> {
    let mut learner = unpack!(req);

    let examples = req.get_examples();
    let xs = unpack_feature_matrix(examples, learner.get_inputs()?)?;
    let ys = unpack_target_vec!(examples, learner.get_targets()?)?;
    learner.train(&xs, &ys)?;

    Db::instance().set("state", learner.get_state())?;

    Ok(TrainingResponse::new())
}

fn infer(req: &InferenceRequest) -> Result<InferenceResponse> {
    let learner = unpack!(req);

    let xs = unpack_feature_matrix(req.get_examples(), learner.get_inputs()?)?;
    let preds = learner.infer(&xs)?;

    let mut response = InferenceResponse::new();
    response.set_predictions(pack_proto(vec![
        ("preds".to_string(), Matrix::from(preds)),
    ])?);
    Ok(response)
}
