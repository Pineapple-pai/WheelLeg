param(
    [Parameter(Mandatory = $true)]
    [string]$SourceAssembly,
    [string]$OutputAssembly = ".\model_source\uz05\solidworks_repaired\chassis.cad_repaired.SLDASM"
)

$ErrorActionPreference = "Stop"
$source = [IO.Path]::GetFullPath($SourceAssembly)
$output = [IO.Path]::GetFullPath($OutputAssembly)
if (-not [IO.File]::Exists($source)) {
    throw "Source assembly not found: $source"
}
if ([IO.File]::Exists($output)) {
    throw "Refusing to overwrite an existing repaired assembly: $output"
}
[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($output)) | Out-Null

$interopRoot = "D:\Program Files\SOLIDWORKS Corp\SOLIDWORKS\api\redist"
$sldworksInterop = Join-Path $interopRoot "SolidWorks.Interop.sldworks.dll"
$swconstInterop = Join-Path $interopRoot "SolidWorks.Interop.swconst.dll"
Add-Type -Path $sldworksInterop
Add-Type -Path $swconstInterop

Add-Type -ReferencedAssemblies $sldworksInterop, $swconstInterop -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;

public static class Uz05ChassisMateRepair
{
    private sealed class RepairSpec
    {
        public string MateName;
        public string LegComponent;
        public string GearboxComponent;
    }

    private static readonly RepairSpec[] Repairs = new[]
    {
        new RepairSpec { MateName = "\u540c\u5fc3259", LegComponent = "\u955c\u5411\u5c0f\u817f-\u65391.15-2", GearboxComponent = "ASM\u51cf\u901f\u7bb1-1/3508\u5c41\u5c41-1" },
        new RepairSpec { MateName = "\u540c\u5fc3260", LegComponent = "\u5c0f\u817f-\u65391.15-1", GearboxComponent = "ASM\u51cf\u901f\u7bb1-2/3508\u5c41\u5c41-1" },
    };

    public static string Repair(object applicationObject, string sourcePath, string outputPath)
    {
        ISldWorks application = (ISldWorks)applicationObject;
        IModelDoc2 active = application.ActiveDoc as IModelDoc2;
        if (active != null && string.Equals(active.GetPathName(), sourcePath, StringComparison.OrdinalIgnoreCase))
            application.CloseDoc(active.GetTitle());

        int openErrors = 0;
        int openWarnings = 0;
        IModelDoc2 model = application.OpenDoc6(
            sourcePath,
            (int)swDocumentTypes_e.swDocASSEMBLY,
            (int)swOpenDocOptions_e.swOpenDocOptions_Silent,
            "",
            ref openErrors,
            ref openWarnings);
        if (model == null)
            throw new InvalidOperationException("Could not open assembly copy; error=" + openErrors);

        IAssemblyDoc assembly = (IAssemblyDoc)model;
        assembly.ResolveAllLightWeightComponents(false);
        Dictionary<string, double[]> transformsBefore = CaptureTransforms(
            assembly,
            new[] { Repairs[0].LegComponent, Repairs[1].LegComponent });
        List<string> results = new List<string>();
        foreach (RepairSpec spec in Repairs)
            results.Add(RepairMate(model, assembly, spec));

        bool rebuildReportedSuccess = model.ForceRebuild3(false);
        if (model.Extension.NeedsRebuild)
            model.EditRebuild3();
        if (model.Extension.NeedsRebuild)
            throw new InvalidOperationException("SolidWorks still requires a rebuild after mate removal.");
        ValidateTransformsUnchanged(assembly, transformsBefore, 1e-8);

        List<string> activeErrors = GetActiveMateErrors(model);
        if (activeErrors.Count != 0)
            throw new InvalidOperationException("Active mate errors remain: " + string.Join(", ", activeErrors));

        int saveErrors = 0;
        int saveWarnings = 0;
        if (!model.Extension.SaveAs(
            outputPath,
            (int)swSaveAsVersion_e.swSaveAsCurrentVersion,
            (int)swSaveAsOptions_e.swSaveAsOptions_Silent,
            null,
            ref saveErrors,
            ref saveWarnings))
            throw new InvalidOperationException(
                "Could not save repaired assembly; errors=" + saveErrors + ", warnings=" + saveWarnings);

        return string.Join("; ", results) +
            "; active_mate_errors=0" +
            "; rebuild_reported_success=" + rebuildReportedSuccess +
            "; open_errors=" + openErrors +
            "; open_warnings=" + openWarnings +
            "; save_errors=" + saveErrors +
            "; save_warnings=" + saveWarnings;
    }

    private static string RepairMate(IModelDoc2 model, IAssemblyDoc assembly, RepairSpec spec)
    {
        IFeature feature = FindMateFeature(model, spec.MateName);
        if (feature == null)
            throw new InvalidOperationException("Mate not found: " + spec.MateName);
        IMate2 oldMate = feature.GetSpecificFeature2() as IMate2;
        if (oldMate == null || oldMate.GetMateEntityCount() != 2)
            throw new InvalidOperationException("Unexpected mate structure: " + spec.MateName);

        IMateEntity2 legEntity = oldMate.MateEntity(0);
        IMateEntity2 gearboxEntity = oldMate.MateEntity(1);
        double[] legTarget = legEntity.EntityParams as double[];
        double[] gearboxTarget = gearboxEntity.EntityParams as double[];
        if (legTarget == null || gearboxTarget == null)
            throw new InvalidOperationException("Mate has no cached geometry: " + spec.MateName);

        IComponent2 leg = FindComponent(assembly, spec.LegComponent);
        IFace2 legFace = FindUniqueCylinderFace(leg, legTarget, 0.0005, spec.MateName + " leg");
        IEntity safeLegFace = ((IEntity)legFace).GetSafeEntity();
        IEntity gearboxFace = gearboxEntity.Reference as IEntity;
        IEntity safeGearboxFace = gearboxFace == null ? null : gearboxFace.GetSafeEntity();
        if (safeLegFace == null || safeGearboxFace == null)
            throw new InvalidOperationException("Could not preserve mate faces for " + spec.MateName);

        model.ClearSelection2(true);
        if (!feature.Select2(false, 0) || !model.Extension.DeleteSelection2(0))
            throw new InvalidOperationException("Could not delete invalid mate: " + spec.MateName);

        return spec.MateName + "_removed_as_redundant";
    }

    private static IFeature FindMateFeature(IModelDoc2 model, string name)
    {
        IFeature feature = model.FirstFeature() as IFeature;
        while (feature != null)
        {
            if (feature.GetTypeName2() == "MateGroup")
            {
                IFeature child = feature.GetFirstSubFeature() as IFeature;
                while (child != null)
                {
                    if (child.Name == name)
                        return child;
                    child = child.GetNextSubFeature() as IFeature;
                }
            }
            feature = feature.GetNextFeature() as IFeature;
        }
        return null;
    }

    private static IComponent2 FindComponent(IAssemblyDoc assembly, string name)
    {
        object[] components = assembly.GetComponents(false) as object[];
        if (components != null)
            foreach (object item in components)
            {
                IComponent2 component = item as IComponent2;
                if (component != null && component.Name2 == name)
                    return component;
            }
        throw new InvalidOperationException("Component not found: " + name);
    }

    private static Dictionary<string, double[]> CaptureTransforms(IAssemblyDoc assembly, string[] names)
    {
        Dictionary<string, double[]> result = new Dictionary<string, double[]>();
        foreach (string name in names)
        {
            IComponent2 component = FindComponent(assembly, name);
            result[name] = (double[])component.Transform2.ArrayData;
        }
        return result;
    }

    private static void ValidateTransformsUnchanged(
        IAssemblyDoc assembly,
        Dictionary<string, double[]> expected,
        double tolerance)
    {
        foreach (KeyValuePair<string, double[]> item in expected)
        {
            double[] actual = (double[])FindComponent(assembly, item.Key).Transform2.ArrayData;
            double maximum = 0.0;
            for (int index = 0; index < actual.Length; index++)
                maximum = Math.Max(maximum, Math.Abs(actual[index] - item.Value[index]));
            if (maximum > tolerance)
                throw new InvalidOperationException(
                    "Component moved after mate repair: " + item.Key +
                    ", max transform delta=" + maximum.ToString("G17", CultureInfo.InvariantCulture));
        }
    }

    private static IFace2 FindUniqueCylinderFace(
        IComponent2 component,
        double[] target,
        double radiusTolerance,
        string label)
    {
        double[] transform = component.Transform2.ArrayData as double[];
        object[] bodies = component.GetBodies2((int)swBodyType_e.swSolidBody) as object[];
        List<IFace2> candidates = new List<IFace2>();
        if (bodies != null)
            foreach (object bodyObject in bodies)
            {
                IBody2 body = bodyObject as IBody2;
                object[] faces = body == null ? null : body.GetFaces() as object[];
                if (faces == null)
                    continue;
                foreach (object faceObject in faces)
                {
                    IFace2 face = faceObject as IFace2;
                    ISurface surface = face == null ? null : face.GetSurface() as ISurface;
                    if (surface == null || !surface.IsCylinder())
                        continue;
                    double[] cylinder = surface.CylinderParams as double[];
                    double[] point = Transform(cylinder, transform, false);
                    double[] direction = Transform(new[] { cylinder[3], cylinder[4], cylinder[5] }, transform, true);
                    Normalize(direction);
                    double[] targetDirection = new[] { target[3], target[4], target[5] };
                    Normalize(targetDirection);
                    double parallel = Math.Abs(Dot(direction, targetDirection));
                    double lineError = LineDistance(point, new[] { target[0], target[1], target[2] }, targetDirection);
                    if (parallel > 0.999999 && lineError < 1e-6 && Math.Abs(cylinder[6] - target[6]) <= radiusTolerance)
                        candidates.Add(face);
                }
            }
        if (candidates.Count != 1)
            throw new InvalidOperationException(
                label + " expected exactly one matching cylindrical face, found " + candidates.Count);
        return candidates[0];
    }

    private static double[] Transform(double[] value, double[] transform, bool vector)
    {
        double x = value[0], y = value[1], z = value[2];
        return new[]
        {
            transform[0] * x + transform[3] * y + transform[6] * z + (vector ? 0.0 : transform[9]),
            transform[1] * x + transform[4] * y + transform[7] * z + (vector ? 0.0 : transform[10]),
            transform[2] * x + transform[5] * y + transform[8] * z + (vector ? 0.0 : transform[11]),
        };
    }

    private static void Normalize(double[] value)
    {
        double length = Math.Sqrt(Dot(value, value));
        for (int index = 0; index < value.Length; index++)
            value[index] /= length;
    }

    private static double Dot(double[] left, double[] right)
    {
        return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
    }

    private static double LineDistance(double[] point, double[] targetPoint, double[] targetDirection)
    {
        double[] delta = new[] { point[0] - targetPoint[0], point[1] - targetPoint[1], point[2] - targetPoint[2] };
        double[] cross = new[]
        {
            delta[1] * targetDirection[2] - delta[2] * targetDirection[1],
            delta[2] * targetDirection[0] - delta[0] * targetDirection[2],
            delta[0] * targetDirection[1] - delta[1] * targetDirection[0],
        };
        return Math.Sqrt(Dot(cross, cross));
    }

    private static List<string> GetActiveMateErrors(IModelDoc2 model)
    {
        List<string> errors = new List<string>();
        IFeature feature = model.FirstFeature() as IFeature;
        while (feature != null)
        {
            if (feature.GetTypeName2() == "MateGroup")
            {
                IFeature child = feature.GetFirstSubFeature() as IFeature;
                while (child != null)
                {
                    if (!child.IsSuppressed() && child.GetSpecificFeature2() is IMate2 && child.GetErrorCode() != 0)
                        errors.Add(child.Name + "=" + child.GetErrorCode().ToString(CultureInfo.InvariantCulture));
                    child = child.GetNextSubFeature() as IFeature;
                }
            }
            feature = feature.GetNextFeature() as IFeature;
        }
        return errors;
    }
}
'@

$application = [Runtime.InteropServices.Marshal]::GetActiveObject("SldWorks.Application")
[Uz05ChassisMateRepair]::Repair($application, $source, $output)
Write-Output "Repaired assembly: $output"
