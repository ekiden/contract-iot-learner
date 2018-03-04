#![feature(use_extern_macros)]

#[macro_use]
extern crate clap;
extern crate futures;
#[macro_use]
extern crate lazy_static;
extern crate protobuf;
extern crate tokio_core;

#[macro_use]
extern crate client_utils;
extern crate ekiden_core_common;
extern crate ekiden_rpc_client;

extern crate iot_learner_api;

use clap::{App, Arg};
use futures::future::Future;

use ekiden_rpc_client::create_client_rpc;
use iot_learner_api::with_api;

with_api! {
    create_client_rpc!(iot, iot_learner_api, api);
}

const USER: &str = "Rusty Lerner";
lazy_static! {
    static ref EXAMPLES: Vec<iot::Example> = {
        let mut ds_proto = std::fs::File::open(
                concat!(env!("CARGO_MANIFEST_DIR"), "/../iot_data.pb")
            )
            .expect("Unable to open dataset.");
        let examples_proto: iot::Examples = protobuf::parse_from_reader(&mut ds_proto)
            .expect("Unable to parse dataset.");
        let mut examples = examples_proto.get_examples().to_vec();
        examples.resize(32, iot::Example::new());
        examples
    };
}

fn init<Backend>(client: &mut iot::Client<Backend>, _runs: usize, _threads: usize)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
    let _create_res = client
        .create({
            let mut req = iot::CreateRequest::new();
            req.set_requester(USER.to_string());
            let inputs = vec!["tin", "tin_a1", "tin_a2"]
                .into_iter()
                .map(String::from)
                .collect();
            let targets = vec!["next_temp".to_string()];
            req.set_inputs(protobuf::RepeatedField::from_vec(inputs));
            req.set_targets(protobuf::RepeatedField::from_vec(targets));
            req
        })
        .wait()
        .expect("error: create");

    let _train_res = client
        .train({
            let mut req = iot::TrainingRequest::new();
            req.set_requester(USER.to_string());
            req.set_examples(protobuf::RepeatedField::from_vec(EXAMPLES.to_vec()));
            req
        })
        .wait()
        .expect("error: train");
}

fn scenario<Backend>(client: &mut iot::Client<Backend>)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
    let _infer_res = client
        .infer({
            let mut req = iot::InferenceRequest::new();
            req.set_requester(USER.to_string());
            req.set_examples(protobuf::RepeatedField::from_slice(
                &EXAMPLES.as_slice()[1..2],
            ));
            req
        })
        .wait()
        .expect("error: infer");
}

fn finalize<Backend>(_client: &mut iot::Client<Backend>, _runs: usize, _threads: usize)
where
    Backend: ekiden_rpc_client::backend::ContractClientBackend,
{
}

fn main() {
    let results = benchmark_client!(iot, init, scenario, finalize);
    results.show();
}
