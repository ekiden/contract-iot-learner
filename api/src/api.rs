rpc_api! {
    metadata {
        name = iot_learner;
        version = "0.1.0";
        client_attestation_required = false;
    }

    rpc create(CreateRequest) -> CreateResponse;

    rpc train(TrainingRequest) -> TrainingResponse;

    rpc infer(InferenceRequest) -> InferenceResponse;
}
