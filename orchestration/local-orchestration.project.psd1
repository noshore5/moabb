@{
    SchemaVersion = 1
    ProjectId = 'wct-gnn'
    ContextFragment = 'orchestration/WCTOrchestrationProjectContext.md'
    AdapterScript = 'orchestration/WctLocalOrchestrationAdapter.ps1'
    AdditionalControlFiles = @(
        'orchestration/WctLocalOrchestrationPreflight.py'
    )
    RequiredSettings = @(
        @{
            Name = 'EnvironmentRoot'
            Kind = 'ExistingDirectory'
            Access = 'ReadExecute'
            Prompt = 'Shared Python environment root'
        }
        @{
            Name = 'DataRoot'
            Kind = 'ExistingDirectory'
            Access = 'ReadOnly'
            Prompt = 'Existing shared EEG dataset root'
        }
        @{
            Name = 'CoherentMultiplexRoot'
            Kind = 'ExistingDirectory'
            Access = 'ReadOnly'
            Prompt = 'Coherent_Multiplex source root'
        }
        @{
            Name = 'GpuVisibleDevices'
            Kind = 'String'
            Access = 'ReadOnly'
            Prompt = 'CUDA_VISIBLE_DEVICES value used for queued WCT GPU runs'
        }
    )
}
